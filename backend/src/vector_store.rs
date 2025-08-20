use backend::{Review, SearchResult};
use anyhow::{Result, anyhow};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tokio::sync::RwLock;
use sha2::{Sha256, Digest};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use std::sync::Arc;

// Simple vector index entry
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default = "default_timestamp")]
    pub timestamp: u64,
    #[serde(default = "default_version")]
    pub version: u32,
}

fn default_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn default_version() -> u32 {
    1
}

// SPFresh-inspired search statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub total_searches: usize,
    pub total_search_time_ms: usize,
    pub cache_hits: usize,
    pub index_reads: usize,
    pub vectors_compared: usize,
}

// SPFresh-inspired dynamic index configuration
#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub max_cache_size: usize,
    pub search_k: usize,
    pub internal_result_num: usize,
    pub similarity_threshold: f32,
    pub enable_quantization: bool,
    pub batch_size: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 10000,
            search_k: 10,
            internal_result_num: 64,
            similarity_threshold: 0.0, // Filter out negative similarities  
            enable_quantization: false,
            batch_size: 100,
        }
    }
}

// Query result cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    results: Vec<SearchResult>,
    timestamp: Instant,
    query_hash: u64,
}

// Simple mock embedding service (for demonstration)
// In production, this would be replaced with a real embedding model
pub struct MockEmbeddingModel {
    dimension: usize,
}

impl MockEmbeddingModel {
    pub fn new() -> Self {
        Self { dimension: 384 } // Simulate 384-dimensional embeddings like sentence transformers
    }

    // Create a deterministic embedding based on text content
    pub fn embed(&self, text: &str) -> Vec<f32> {
        // Normalize and clean text
        let cleaned_text = text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>();
        
        // Create hash for deterministic seed
        let mut hasher = Sha256::new();
        hasher.update(cleaned_text.as_bytes());
        let hash = hasher.finalize();
        
        // Use hash as seed for random number generator
        let seed_bytes: [u8; 32] = hash.into();
        let mut rng = StdRng::from_seed(seed_bytes);
        
        // Generate base vector from hash
        let mut vector = vec![0.0f32; self.dimension];
        for i in 0..self.dimension {
            vector[i] = rng.gen_range(-1.0..1.0);
        }
        
        // Add semantic features based on text characteristics
        self.add_semantic_features(&mut vector, &cleaned_text);
        
        // Normalize vector
        self.normalize_vector(&mut vector);
        
        vector
    }
    
    fn add_semantic_features(&self, vector: &mut Vec<f32>, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f32;
        
        // Add features based on word count
        if self.dimension > 10 {
            vector[0] += (word_count / 100.0).min(1.0);
        }
        
        // Add features for common semantic categories
        let positive_words = ["good", "great", "excellent", "amazing", "love", "perfect", "best"];
        let negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible"];
        let tech_words = ["battery", "camera", "screen", "performance", "fast", "slow", "memory"];
        
        let positive_score = self.count_word_matches(&words, &positive_words) as f32;
        let negative_score = self.count_word_matches(&words, &negative_words) as f32;
        let tech_score = self.count_word_matches(&words, &tech_words) as f32;
        
        if self.dimension > 50 {
            vector[1] += (positive_score / word_count).min(1.0);
            vector[2] -= (negative_score / word_count).min(1.0);
            vector[3] += (tech_score / word_count).min(1.0);
        }
        
        // Add length-based features
        if self.dimension > 100 {
            vector[4] += (text.len() as f32 / 1000.0).min(1.0);
        }
    }
    
    fn count_word_matches(&self, words: &[&str], patterns: &[&str]) -> usize {
        words.iter()
            .filter(|word| patterns.iter().any(|pattern| word.contains(pattern)))
            .count()
    }
    
    fn normalize_vector(&self, vector: &mut Vec<f32>) {
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for x in vector.iter_mut() {
                *x /= magnitude;
            }
        }
    }
}

// File-based vector store (implementing basic functionality inspired by SPFresh)
pub struct VectorStore {
    embedding_model: MockEmbeddingModel,
    index_path: String,
    metadata_path: String,
    vectors: RwLock<Vec<VectorEntry>>,
    metadata: RwLock<Vec<Review>>,
    // SPFresh-inspired features
    config: IndexConfig,
    stats: RwLock<SearchStats>,
    query_cache: RwLock<HashMap<u64, CacheEntry>>,
}

impl VectorStore {
    pub async fn new(data_dir: &str) -> Result<Self> {
        // Create data directory if it doesn't exist
        std::fs::create_dir_all(data_dir)?;
        
        let index_path = format!("{}/reviews.index", data_dir);
        let metadata_path = format!("{}/reviews.jsonl", data_dir);
        
        // Initialize mock embedding model
        let model = MockEmbeddingModel::new();

        let store = Self {
            embedding_model: model,
            index_path: index_path.clone(),
            metadata_path: metadata_path.clone(),
            vectors: RwLock::new(Vec::new()),
            metadata: RwLock::new(Vec::new()),
            config: IndexConfig::default(),
            stats: RwLock::new(SearchStats::default()),
            query_cache: RwLock::new(HashMap::new()),
        };

        // Load existing data
        store.load_from_disk().await?;
        
        Ok(store)
    }

    // Create embedding from text
    pub fn create_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(anyhow!("Cannot create embedding from empty text"));
        }
        Ok(self.embedding_model.embed(text))
    }

    // Add a review to the store (append-only)
    pub async fn add_review(&self, review: Review) -> Result<()> {
        // Create embedding from review title + body
        let text = format!("{} {}", review.review_title, review.review_body);
        let vector = self.create_embedding(&text)?;
        
        let vector_entry = VectorEntry {
            id: review.id.clone(),
            vector,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            version: 1,
        };

        // Add to in-memory store
        {
            let mut vectors = self.vectors.write().await;
            let mut metadata = self.metadata.write().await;
            
            vectors.push(vector_entry.clone());
            metadata.push(review.clone());
        }

        // Append to disk
        self.append_to_disk(&vector_entry, &review).await?;
        
        Ok(())
    }

    // Bulk add reviews
    pub async fn add_reviews_bulk(&self, reviews: Vec<Review>) -> Result<usize> {
        let mut added_count = 0;
        
        for review in reviews {
            if let Ok(_) = self.add_review(review).await {
                added_count += 1;
            }
        }
        
        Ok(added_count)
    }

    // Search for similar reviews
    // Enhanced search with SPFresh-inspired features
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let start_time = Instant::now();
        
        // Create query hash for caching
        let query_hash = self.hash_query(query);
        
        // Check cache first
        if let Some(cached_results) = self.check_cache(query_hash).await {
            self.update_stats_cache_hit().await;
            return Ok(cached_results.into_iter().take(limit).collect());
        }

        // Create embedding for the query
        let query_vector = self.create_embedding(query)?;
        let query_array = Array1::from_vec(query_vector);
        
        // Use internal_result_num for more comprehensive search (SPFresh-inspired)
        let internal_limit = self.config.internal_result_num.max(limit * 2);
        
        let vectors = self.vectors.read().await;
        let metadata = self.metadata.read().await;
        
        let mut results = Vec::new();
        let mut vectors_compared = 0;
        
        // Calculate multiple similarity scores for each vector
        for (i, vector_entry) in vectors.iter().enumerate() {
            let vector_array = Array1::from_vec(vector_entry.vector.clone());
            
            // Calculate different similarity scores
            let combined_similarity = cosine_similarity(&query_array, &vector_array);
            
            vectors_compared += 1;
            
            // Also check for direct text matches to boost relevance
            if let Some(review) = metadata.get(i) {
                let mut final_similarity = combined_similarity;
                
                // Boost score for exact keyword matches (case-insensitive)
                let query_lower = query.to_lowercase();
                let title_lower = review.review_title.to_lowercase();
                let body_lower = review.review_body.to_lowercase();
                
                // Check each word in the query for matches
                let query_words: Vec<&str> = query_lower.split_whitespace().collect();
                let mut title_matches = 0;
                let mut body_matches = 0;
                
                for word in &query_words {
                    if title_lower.contains(word) {
                        title_matches += 1;
                    }
                    if body_lower.contains(word) {
                        body_matches += 1;
                    }
                }
                
                // Apply boosting based on matches
                if title_matches > 0 {
                    final_similarity += 0.15 * (title_matches as f32 / query_words.len() as f32); // Title match boost
                }
                
                if body_matches > 0 {
                    final_similarity += 0.1 * (body_matches as f32 / query_words.len() as f32); // Body match boost
                }
                
                // Apply similarity threshold filter (SPFresh-inspired)
                if final_similarity >= self.config.similarity_threshold {
                    results.push(SearchResult {
                        review: review.clone(),
                        similarity_score: final_similarity,
                    });
                }
            }
        }
        
        // Sort by similarity (descending) and take internal results first
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        results.truncate(internal_limit);
        
        // Cache the internal results for future queries
        self.cache_results(query_hash, results.clone()).await;
        
        // Update statistics
        self.update_stats(start_time.elapsed().as_millis() as usize, vectors_compared).await;
        
        // Return only the requested limit
        Ok(results.into_iter().take(limit).collect())
    }

    // SPFresh-inspired helper methods
    fn hash_query(&self, query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.to_lowercase().trim().hash(&mut hasher);
        hasher.finish()
    }

    async fn check_cache(&self, query_hash: u64) -> Option<Vec<SearchResult>> {
        let cache = self.query_cache.read().await;
        if let Some(entry) = cache.get(&query_hash) {
            // Check if cache entry is still valid (within 5 minutes)
            if entry.timestamp.elapsed().as_secs() < 300 {
                return Some(entry.results.clone());
            }
        }
        None
    }

    async fn cache_results(&self, query_hash: u64, results: Vec<SearchResult>) {
        let mut cache = self.query_cache.write().await;
        
        // Implement cache size limit
        if cache.len() >= self.config.max_cache_size {
            // Remove oldest entries (simple FIFO)
            let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 2).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
        
        cache.insert(query_hash, CacheEntry {
            results,
            timestamp: Instant::now(),
            query_hash,
        });
    }

    async fn update_stats_cache_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.cache_hits += 1;
        stats.total_searches += 1;
    }

    async fn update_stats(&self, search_time_ms: usize, vectors_compared: usize) {
        let mut stats = self.stats.write().await;
        stats.total_searches += 1;
        stats.total_search_time_ms += search_time_ms;
        stats.index_reads += 1;
        stats.vectors_compared += vectors_compared;
    }

    // Get current search statistics
    pub async fn get_stats(&self) -> SearchStats {
        self.stats.read().await.clone()
    }

    // Load existing data from disk
    async fn load_from_disk(&self) -> Result<()> {
        // Load vectors from index file
        if Path::new(&self.index_path).exists() {
            let file = File::open(&self.index_path)?;
            let reader = BufReader::new(file);
            
            let mut vectors = self.vectors.write().await;
            for line in reader.lines() {
                let line = line?;
                if let Ok(vector_entry) = serde_json::from_str::<VectorEntry>(&line) {
                    vectors.push(vector_entry);
                }
            }
        }

        // Load metadata from jsonl file
        if Path::new(&self.metadata_path).exists() {
            let file = File::open(&self.metadata_path)?;
            let reader = BufReader::new(file);
            
            let mut metadata = self.metadata.write().await;
            for line in reader.lines() {
                let line = line?;
                if let Ok(review) = serde_json::from_str::<Review>(&line) {
                    metadata.push(review);
                }
            }
        }

        Ok(())
    }

    // Append new data to disk files
    async fn append_to_disk(&self, vector_entry: &VectorEntry, review: &Review) -> Result<()> {
        // Append vector to index file
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.index_path)?;
            
            let vector_json = serde_json::to_string(vector_entry)?;
            writeln!(file, "{}", vector_json)?;
        }

        // Append metadata to jsonl file
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.metadata_path)?;
            
            let review_json = serde_json::to_string(review)?;
            writeln!(file, "{}", review_json)?;
        }

        Ok(())
    }

    // Get total count of stored reviews
    pub async fn count(&self) -> usize {
        let metadata = self.metadata.read().await;
        metadata.len()
    }

    // SPFresh-inspired additional methods
    
    // Get index configuration
    pub fn get_config(&self) -> &IndexConfig {
        &self.config
    }
    
    // Update search configuration (dynamic reconfiguration)
    pub async fn update_config(&mut self, new_config: IndexConfig) {
        self.config = new_config;
        // Clear cache when configuration changes
        let mut cache = self.query_cache.write().await;
        cache.clear();
    }
    
    // Force cache compaction (remove expired entries)
    pub async fn compact_cache(&self) {
        let mut cache = self.query_cache.write().await;
        let now = Instant::now();
        cache.retain(|_, entry| now.duration_since(entry.timestamp).as_secs() < 300);
    }
    
    // Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.query_cache.read().await;
        let stats = self.stats.read().await;
        (cache.len(), stats.cache_hits)
    }
    
    // Batch add reviews (SPFresh-inspired bulk operations)
    pub async fn add_reviews_batch(&self, reviews: Vec<Review>) -> Result<usize> {
        let mut added_count = 0;
        
        for review in reviews {
            if let Ok(()) = self.add_review(review).await {
                added_count += 1;
            }
        }
        
        // Force cache clear after bulk operations
        let mut cache = self.query_cache.write().await;
        cache.clear();
        
        Ok(added_count)
    }
    
    // Get memory usage statistics
    pub async fn get_memory_stats(&self) -> (usize, usize, usize) {
        let vectors = self.vectors.read().await;
        let metadata = self.metadata.read().await;
        let cache = self.query_cache.read().await;
        
        let vector_memory = vectors.len() * vectors.first()
            .map(|v| v.vector.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let metadata_memory = metadata.len() * std::mem::size_of::<Review>();
        let cache_memory = cache.len() * std::mem::size_of::<CacheEntry>();
        
        (vector_memory, metadata_memory, cache_memory)
    }
}

// Helper function to calculate cosine similarity
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = (a.dot(a)).sqrt();
    let norm_b = (b.dot(b)).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
