use axum::{
    extract::{rejection::JsonRejection, State},
    http::{Method, StatusCode},
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde_json::json;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};
use uuid::Uuid;
use chrono::Utc;


use backend::{Review, ReviewInput, SearchQuery, SearchResponse, BulkInsertRequest, InsertResponse};

mod vector_store;
use vector_store::VectorStore;


type AppState = Arc<VectorStore>;

// Custom JSON extractor that returns JSON error responses
struct JsonExtractor<T>(pub T);

#[axum::async_trait]
impl<T, S> axum::extract::FromRequest<S> for JsonExtractor<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request(
        req: axum::extract::Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(JsonExtractor(value)),
            Err(rejection) => {
                let error_message = match rejection {
                    JsonRejection::JsonDataError(err) => format!("Invalid JSON data: {}", err),
                    JsonRejection::JsonSyntaxError(err) => format!("JSON syntax error: {}", err),
                    JsonRejection::MissingJsonContentType(_) => "Missing Content-Type: application/json header".to_string(),
                    _ => "Invalid JSON request".to_string(),
                };
                
                Err((
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({
                        "success": false,
                        "error": error_message
                    }))
                ))
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
   
    let data_path = if std::path::Path::new("../data").exists() {
        "../data"
    } else {
        "./data"  
    };
    let vector_store = VectorStore::new(data_path).await?;
    let state = Arc::new(vector_store);


    let app = Router::new()
        .route("/", get(index))
        .route("/api/reviews", post(add_review))
        .route("/api/reviews/bulk", post(add_reviews_bulk))
        .route("/api/search", post(search_reviews))
        .route("/api/stats", get(get_stats))

        .route("/api/stats/search", get(get_search_stats))
        .route("/api/stats/cache", get(get_cache_stats))
        .route("/api/stats/memory", get(get_memory_stats))
        .route("/api/config", get(get_config))
        .route("/api/cache/compact", post(compact_cache))
        .nest_service("/static", ServeDir::new("./static"))
        .layer(
            ServiceBuilder::new()
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods([Method::GET, Method::POST])
                        .allow_headers(Any),
                )
        )
        .with_state(state);

    // Run our app with hyper
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000")
        .await
        .unwrap();
    
    println!("🚀 Review Semantic Search Server running on http://127.0.0.1:8000");
    println!("📁 Data directory: {}", data_path);
    println!("🔍 Ready for semantic search operations!");
    axum::serve(listener, app).await.unwrap();
    
    Ok(())
}


async fn index() -> impl IntoResponse {
    Html(include_str!("../static/index.html"))
}

// API endpoint to add a single review
async fn add_review(
    State(store): State<AppState>, 
    JsonExtractor(review_input): JsonExtractor<ReviewInput>
) -> impl IntoResponse {
    println!("Received review request: {:?}", review_input);
    

    let review = Review {
        id: if review_input.id.is_empty() { 
            Uuid::new_v4().to_string() 
        } else { 
            review_input.id 
        },
        review_title: review_input.review_title.trim().to_string(),
        review_body: review_input.review_body.trim().to_string(),
        product_id: review_input.product_id.trim().to_string(),
        review_rating: review_input.review_rating,
        created_at: review_input.created_at.unwrap_or_else(|| Utc::now()),
    };

    println!("After validation: {:?}", review);

    if review.review_title.is_empty() || review.review_body.is_empty() {
        return (
            StatusCode::BAD_REQUEST, 
            Json(json!({
                "success": false,
                "error": "Review title and body are required and cannot be empty"
            }))
        );
    }

    if review.product_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST, 
            Json(json!({
                "success": false,
                "error": "Product ID is required and cannot be empty"
            }))
        );
    }

    if review.review_rating > 5 || review.review_rating < 1 {
        return (
            StatusCode::BAD_REQUEST, 
            Json(json!({
                "success": false,
                "error": "Review rating must be between 1 and 5"
            }))
        );
    }

    match store.add_review(review.clone()).await {
        Ok(_) => {
            println!("Successfully added review: {}", review.id);
            (
                StatusCode::CREATED, 
                Json(json!({
                    "success": true,
                    "message": "Review added successfully",
                    "review": review
                }))
            )
        },
        Err(e) => {
            let error_msg = e.to_string();
            eprintln!("Error adding review: {}", error_msg);
            println!("Full error details: {:?}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR, 
                Json(json!({
                    "success": false,
                    "error": "Failed to add review",
                    "details": error_msg
                }))
            )
        },
    }
}

// API endpoint for bulk insert
async fn add_reviews_bulk(
    State(store): State<AppState>, 
    JsonExtractor(request): JsonExtractor<BulkInsertRequest>
) -> impl IntoResponse {
    let mut processed_reviews = Vec::new();
    
    for review_input in request.reviews {
        // Convert ReviewInput to Review
        let review = Review {
            id: if review_input.id.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                review_input.id
            },
            review_title: review_input.review_title,
            review_body: review_input.review_body,
            product_id: review_input.product_id,
            review_rating: review_input.review_rating,
            created_at: review_input.created_at.unwrap_or_else(|| Utc::now()),
        };
        
        // Basic validation
        if !review.review_title.is_empty() && !review.review_body.is_empty() 
           && review.review_rating >= 1 && review.review_rating <= 5 {
            processed_reviews.push(review);
        }
    }

    match store.add_reviews_bulk(processed_reviews.clone()).await {
        Ok(inserted_count) => (
            StatusCode::CREATED, 
            Json(json!(InsertResponse {
                success: true,
                inserted_count,
                message: format!("Successfully inserted {} reviews", inserted_count)
            }))
        ),
        Err(e) => {
            let error_msg = e.to_string();
            eprintln!("Error in bulk insert: {}", error_msg);
            (
                StatusCode::INTERNAL_SERVER_ERROR, 
                Json(json!(InsertResponse {
                    success: false,
                    inserted_count: 0,
                    message: "Failed to insert reviews".to_string()
                }))
            )
        },
    }
}

// API endpoint for semantic search
async fn search_reviews(
    State(store): State<AppState>,
    JsonExtractor(search_query): JsonExtractor<SearchQuery>
) -> impl IntoResponse {
    if search_query.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Search query cannot be empty"}))
        );
    }

    let limit = search_query.limit.unwrap_or(10).min(100);

    match store.search(&search_query.query, limit).await {
        Ok(results) => {
            let response = SearchResponse {
                total_found: results.len(),
                query: search_query.query.clone(),
                results,
            };
            (StatusCode::OK, Json(json!(response)))
        },
        Err(e) => {
            let error_msg = e.to_string();
            eprintln!("Search error: {}", error_msg);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "success": false,
                    "error": "Search failed",
                    "details": error_msg
                }))
            )
        },
    }
}

// API endpoint to get stats
async fn get_stats(State(store): State<AppState>) -> impl IntoResponse {
    let total_reviews = store.count().await;
    
    Json(json!({
        "total_reviews": total_reviews,
        "status": "operational",
        "data_directory": "./data"
    }))
}

// Get detailed search statistics
async fn get_search_stats(State(store): State<AppState>) -> impl IntoResponse {
    let stats = store.get_stats().await;
    
    Json(json!({
        "search_stats": {
            "total_searches": stats.total_searches,
            "total_search_time_ms": stats.total_search_time_ms,
            "average_search_time_ms": if stats.total_searches > 0 {
                stats.total_search_time_ms / stats.total_searches
            } else { 0 },
            "cache_hits": stats.cache_hits,
            "cache_hit_rate": if stats.total_searches > 0 {
                (stats.cache_hits as f64 / stats.total_searches as f64 * 100.0).round()
            } else { 0.0 },
            "index_reads": stats.index_reads,
            "vectors_compared": stats.vectors_compared,
            "avg_vectors_per_search": if stats.index_reads > 0 {
                stats.vectors_compared / stats.index_reads
            } else { 0 }
        }
    }))
}

// Get cache statistics
async fn get_cache_stats(State(store): State<AppState>) -> impl IntoResponse {
    let (cache_size, cache_hits) = store.get_cache_stats().await;
    let config = store.get_config();
    
    Json(json!({
        "cache_stats": {
            "current_size": cache_size,
            "max_size": config.max_cache_size,
            "utilization": (cache_size as f64 / config.max_cache_size as f64 * 100.0).round(),
            "total_hits": cache_hits
        }
    }))
}

// Get memory usage statistics
async fn get_memory_stats(State(store): State<AppState>) -> impl IntoResponse {
    let (vector_memory, metadata_memory, cache_memory) = store.get_memory_stats().await;
    let total_memory = vector_memory + metadata_memory + cache_memory;
    
    Json(json!({
        "memory_stats": {
            "vector_memory_bytes": vector_memory,
            "metadata_memory_bytes": metadata_memory,
            "cache_memory_bytes": cache_memory,
            "total_memory_bytes": total_memory,
            "vector_memory_mb": (vector_memory as f64 / 1024.0 / 1024.0).round(),
            "metadata_memory_mb": (metadata_memory as f64 / 1024.0 / 1024.0).round(),
            "cache_memory_mb": (cache_memory as f64 / 1024.0 / 1024.0).round(),
            "total_memory_mb": (total_memory as f64 / 1024.0 / 1024.0).round()
        }
    }))
}

// Get current index configuration
async fn get_config(State(store): State<AppState>) -> impl IntoResponse {
    let config = store.get_config();
    
    Json(json!({
        "config": {
            "max_cache_size": config.max_cache_size,
            "search_k": config.search_k,
            "internal_result_num": config.internal_result_num,
            "similarity_threshold": config.similarity_threshold,
            "enable_quantization": config.enable_quantization,
            "batch_size": config.batch_size
        }
    }))
}

// Force cache compaction
async fn compact_cache(State(store): State<AppState>) -> impl IntoResponse {
    store.compact_cache().await;
    let (cache_size, _) = store.get_cache_stats().await;
    
    Json(json!({
        "success": true,
        "message": "Cache compacted successfully",
        "current_cache_size": cache_size
    }))
}