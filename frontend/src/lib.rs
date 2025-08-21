use leptos::*;
use leptos_meta::*;
use leptos_router::*;
use serde::{Deserialize, Serialize};


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
pub struct SearchStats {
    pub total_searches: usize,
    pub total_search_time_ms: usize,
    pub average_search_time_ms: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub index_reads: usize,
    pub vectors_compared: usize,
    pub avg_vectors_per_search: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CacheStats {
    pub current_size: usize,
    pub max_size: usize,
    pub utilization: f64,
    pub total_hits: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryStats {
    pub vector_memory_bytes: usize,
    pub metadata_memory_bytes: usize,
    pub cache_memory_bytes: usize,
    pub total_memory_bytes: usize,
    pub total_memory_mb: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Review {
    pub id: String,
    pub review_title: String,
    pub review_body: String,
    pub product_id: String,
    pub review_rating: u8,
    pub created_at: String,
}


async fn call_search_api(query: SearchQuery) -> Result<SearchResponse, String> {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};
    
    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    
    let body = serde_json::to_string(&query)
        .map_err(|e| format!("Serialization error: {}", e))?;
    opts.set_body(&JsValue::from_str(&body));
    
    let request = Request::new_with_str_and_init("http://127.0.0.1:8000/api/search", &opts)
        .map_err(|e| format!("Request creation error: {:?}", e))?;
    
    request.headers().set("Content-Type", "application/json")
        .map_err(|e| format!("Header error: {:?}", e))?;
    
    let window = web_sys::window().ok_or("No window object")?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await
        .map_err(|e| format!("Fetch error: {:?}", e))?;
    
    let resp: Response = resp_value.dyn_into()
        .map_err(|e| format!("Response conversion error: {:?}", e))?;
    
    let text = JsFuture::from(resp.text().map_err(|e| format!("Text error: {:?}", e))?)
        .await
        .map_err(|e| format!("Text await error: {:?}", e))?;
    
    let text_str = text.as_string().ok_or("No text string")?;
    
    serde_json::from_str::<SearchResponse>(&text_str)
        .map_err(|e| format!("Parse error: {}", e))
}

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/frontend.css"/>
        <Stylesheet id="custom" href="/static/style.css"/>
        <Title text="SPFresh-Inspired Vector Search"/>

        <Router>
            <main>
                <Routes>
                    <Route path="" view=HomePage/>
                    <Route path="/search" view=SearchPage/>
                    <Route path="/stats" view=StatsPage/>
                </Routes>
            </main>
        </Router>
    }
}

#[component]
fn HomePage() -> impl IntoView {
    view! {
        <div class="container">
            <h1>"SPFresh-Inspired Vector Search Demo"</h1>
            <p>"Welcome to the Rust vector search application with SPFresh-inspired features"</p>
            
            <div class="nav-links">
                <a href="/search" class="nav-button">"🔍 Semantic Search"</a>
                <a href="/stats" class="nav-button">"📊 Performance Stats"</a>
            </div>
            
            <div class="features">
                <h2>"Features Inspired by SPFresh:"</h2>
                <ul>
                    <li>"• Dynamic query caching for improved performance"</li>
                    <li>"• Real-time search statistics and monitoring"</li>
                    <li>"• Configurable similarity thresholds"</li>
                    <li>"• Memory usage optimization"</li>
                    <li>"• Bulk operations support"</li>
                    <li>"• Cache management and compaction"</li>
                </ul>
            </div>
        </div>
    }
}

#[component]
fn SearchPage() -> impl IntoView {
    let (query, set_query) = create_signal(String::new());
    let (results, set_results) = create_signal(Vec::<String>::new());

    let search_action = create_action(move |query: &String| {
        let query = query.clone();
        async move {
            if query.is_empty() {
                set_results.set(vec![]);
                return;
            }

  
            let search_request = SearchQuery {
                query: query.clone(),
                limit: Some(10),
            };

            match call_search_api(search_request).await {
                Ok(response) => {
                    let result_strings: Vec<String> = response.results
                        .into_iter()
                        .map(|result| {
                            format!(
                                "⭐ {} - {} (Score: {:.2})\n{}",
                                result.review.review_rating,
                                result.review.review_title,
                                result.similarity_score,
                                result.review.review_body
                            )
                        })
                        .collect();
                    set_results.set(result_strings);
                },
                Err(e) => {
                    set_results.set(vec![format!("Error: {}", e)]);
                }
            }
        }
    });

    view! {
        <div class="container">
            <h1>"Semantic Search"</h1>
            
            <div class="search-form">
                <input
                    type="text"
                    placeholder="Enter your search query..."
                    prop:value=query
                    on:input=move |ev| {
                        set_query.set(event_target_value(&ev));
                    }
                />
                <button
                    on:click=move |_| {
                        search_action.dispatch(query.get());
                    }
                >
                    "Search"
                </button>
            </div>

            <div class="results">
                <For
                    each=move || results.get()
                    key=|result| result.clone()
                    children=move |result| {
                        view! {
                            <div class="result-item">
                                {result}
                            </div>
                        }
                    }
                />
            </div>
        </div>
    }
}

#[component]
fn StatsPage() -> impl IntoView {
    let (search_stats, set_search_stats) = create_signal(None::<SearchStats>);
    let (cache_stats, set_cache_stats) = create_signal(None::<CacheStats>);
    let (memory_stats, set_memory_stats) = create_signal(None::<MemoryStats>);
    let (is_loading, set_is_loading) = create_signal(false);

    let load_stats = create_action(move |_: &()| async move {
        set_is_loading.set(true);
        
 
        if let Ok(response) = call_api::<(), serde_json::Value>("http://127.0.0.1:8000/api/stats/search", "GET", None).await {
            if let Ok(stats) = serde_json::from_value::<SearchStats>(response["search_stats"].clone()) {
                set_search_stats.set(Some(stats));
            }
        }
        
    
        if let Ok(response) = call_api::<(), serde_json::Value>("http://127.0.0.1:8000/api/stats/cache", "GET", None).await {
            if let Ok(stats) = serde_json::from_value::<CacheStats>(response["cache_stats"].clone()) {
                set_cache_stats.set(Some(stats));
            }
        }
        

        if let Ok(response) = call_api::<(), serde_json::Value>("http://127.0.0.1:8000/api/stats/memory", "GET", None).await {
            if let Ok(stats) = serde_json::from_value::<MemoryStats>(response["memory_stats"].clone()) {
                set_memory_stats.set(Some(stats));
            }
        }
        
        set_is_loading.set(false);
    });

    let compact_cache = create_action(move |_: &()| async move {
        let _ = call_api::<(), serde_json::Value>("http://127.0.0.1:8000/api/cache/compact", "POST", None).await;

        load_stats.dispatch(());
    });


    create_effect(move |_| {
        load_stats.dispatch(());
    });

    view! {
        <div class="container">
            <h1>"SPFresh-Inspired Performance Statistics"</h1>
            
            <div class="stats-controls">
                <button 
                    on:click=move |_| load_stats.dispatch(())
                    disabled=move || is_loading.get()
                >
                    {move || if is_loading.get() { "Loading..." } else { "Refresh Stats" }}
                </button>
                
                <button 
                    on:click=move |_| compact_cache.dispatch(())
                    class="compact-button"
                >
                    "Compact Cache"
                </button>
            </div>


            <div class="stats-section">
                <h2>"🔍 Search Performance"</h2>
                {move || match search_stats.get() {
                    Some(stats) => view! {
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">"Total Searches:"</span>
                                <span class="stat-value">{stats.total_searches}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Average Search Time:"</span>
                                <span class="stat-value">{stats.average_search_time_ms}"ms"</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Cache Hit Rate:"</span>
                                <span class="stat-value">{format!("{:.1}%", stats.cache_hit_rate)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Vectors Compared:"</span>
                                <span class="stat-value">{stats.vectors_compared}</span>
                            </div>
                        </div>
                    }.into_view(),
                    None => view! { <p>"Loading search statistics..."</p> }.into_view()
                }}
            </div>


            <div class="stats-section">
                <h2>"💾 Cache Performance"</h2>
                {move || match cache_stats.get() {
                    Some(stats) => view! {
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">"Cache Size:"</span>
                                <span class="stat-value">{stats.current_size}"/"{ stats.max_size}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Cache Utilization:"</span>
                                <span class="stat-value">{format!("{:.1}%", stats.utilization)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Total Cache Hits:"</span>
                                <span class="stat-value">{stats.total_hits}</span>
                            </div>
                        </div>
                        <div class="progress-bar">
                            <div 
                                class="progress-fill"
                                style=format!("width: {}%", stats.utilization)
                            ></div>
                        </div>
                    }.into_view(),
                    None => view! { <p>"Loading cache statistics..."</p> }.into_view()
                }}
            </div>


            <div class="stats-section">
                <h2>"🧠 Memory Usage"</h2>
                {move || match memory_stats.get() {
                    Some(stats) => view! {
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">"Total Memory:"</span>
                                <span class="stat-value">{format!("{:.2} MB", stats.total_memory_mb)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Vector Data:"</span>
                                <span class="stat-value">{format!("{:.2} MB", stats.vector_memory_bytes as f64 / 1024.0 / 1024.0)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Metadata:"</span>
                                <span class="stat-value">{format!("{:.2} MB", stats.metadata_memory_bytes as f64 / 1024.0 / 1024.0)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">"Cache:"</span>
                                <span class="stat-value">{format!("{:.2} MB", stats.cache_memory_bytes as f64 / 1024.0 / 1024.0)}</span>
                            </div>
                        </div>
                    }.into_view(),
                    None => view! { <p>"Loading memory statistics..."</p> }.into_view()
                }}
            </div>
            
            <div class="nav-back">
                <a href="/">"← Back to Home"</a>
                <a href="/search">"→ Go to Search"</a>
            </div>
        </div>
    }
}


async fn call_api<T: Serialize, R: for<'de> Deserialize<'de>>(
    url: &str, 
    method: &str, 
    body: Option<T>
) -> Result<R, String> {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};
    
    let opts = RequestInit::new();
    opts.set_method(method);
    opts.set_mode(RequestMode::Cors);
    
    let has_body = body.is_some();
    
    if let Some(body_data) = body {
        let body_str = serde_json::to_string(&body_data)
            .map_err(|e| format!("Serialization error: {}", e))?;
        opts.set_body(&JsValue::from_str(&body_str));
    }
    
    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("Request creation error: {:?}", e))?;
    
    if has_body {
        request.headers().set("Content-Type", "application/json")
            .map_err(|e| format!("Header error: {:?}", e))?;
    }
    
    let window = web_sys::window().ok_or("No window object")?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await
        .map_err(|e| format!("Fetch error: {:?}", e))?;
    
    let resp: Response = resp_value.dyn_into()
        .map_err(|e| format!("Response conversion error: {:?}", e))?;
    
    let text = JsFuture::from(resp.text().map_err(|e| format!("Text error: {:?}", e))?)
        .await
        .map_err(|e| format!("Text await error: {:?}", e))?;
    
    let text_str = text.as_string().ok_or("No text string")?;
    
    serde_json::from_str::<R>(&text_str)
        .map_err(|e| format!("Parse error: {}", e))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).expect("error initializing log");
    
    leptos::mount_to_body(App)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn main() {

}
