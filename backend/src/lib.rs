
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Review {
    pub id: String,
    pub review_title: String,
    pub review_body: String,
    pub product_id: String,
    pub review_rating: u8,
    pub created_at: DateTime<Utc>,
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReviewInput {
    #[serde(default)]
    pub id: String,
    pub review_title: String,
    pub review_body: String,
    pub product_id: String,
    pub review_rating: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SearchQuery {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SearchResult {
    pub review: Review,
    pub similarity_score: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_found: usize,
    pub query: String,
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BulkInsertRequest {
    pub reviews: Vec<Review>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InsertResponse {
    pub success: bool,
    pub inserted_count: usize,
    pub message: String,
}