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


#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub total_searches: usize,
    pub total_search_time_ms: usize,
    pub cache_hits: usize,
    pub index_reads: usize,
    pub vectors_compared: usize,
}


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
            similarity_threshold: 0.0,
            enable_quantization: false,
            batch_size: 100,
        }
    }
}


#[derive(Debug, Clone)]
struct CacheEntry {
    results: Vec<SearchResult>,
    timestamp: Instant,
    query_hash: u64,
}

pub struct MockEmbeddingModel {
    dimension: usize,
}

impl MockEmbeddingModel {
    pub fn new() -> Self {
        Self { dimension: 384 }
    }


    pub fn embed(&self, text: &str) -> Vec<f32> {

        let cleaned_text = text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>();
        

        let mut hasher = Sha256::new();
        hasher.update(cleaned_text.as_bytes());
        let hash = hasher.finalize();

        let seed_bytes: [u8; 32] = hash.into();
        let mut rng = StdRng::from_seed(seed_bytes);
        

        let mut vector = vec![0.0f32; self.dimension];
        for i in 0..self.dimension {
            vector[i] = rng.gen_range(-1.0..1.0);
        }

        self.add_semantic_features(&mut vector, &cleaned_text);
        

        self.normalize_vector(&mut vector);
        
        vector
    }
    
    fn add_semantic_features(&self, vector: &mut Vec<f32>, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f32;
        

        if self.dimension > 10 {
            vector[0] += (word_count / 100.0).min(1.0);
        }
        

        let positive_words = [
   
            "good", "great", "excellent", "amazing", "love", "perfect", "best",
       
            "fantastic", "wonderful", "awesome", "beautiful", "fabulous", "delicious",
            "outstanding", "exceptional", "brilliant", "lovely", "superb", "terrific",
            "marvelous", "impressive", "stellar", "incredible", "magnificent", "divine",
 
            "friendly", "attentive", "professional", "warm", "welcoming", "accommodating",
            "helpful", "courteous", "pleasant", "caring", "thoughtful", "considerate",

            "fresh", "tasty", "flavorful", "rich", "smooth", "tender", "crispy",
            "perfectly", "well-prepared", "well-made", "high-quality", "top-notch",

            "cozy", "comfortable", "relaxing", "charming", "elegant", "classy",
            "inviting", "stylish", "sophisticated", "atmospheric", "intimate",

            "recommend", "definitely", "highly", "must-try", "worth", "favorite",
            "gem", "treasure", "hidden", "secret", "special", "unique"
        ];
        
        let negative_words = [

            "bad", "terrible", "awful", "hate", "worst", "horrible",

            "disappointing", "poor", "unpleasant", "rude", "slow", "cold",
            "overpriced", "expensive", "bland", "tasteless", "dry", "soggy",
            "dirty", "messy", "noisy", "crowded", "uncomfortable", "unprofessional"
        ];
        
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

pub struct VectorStore {
    embedding_model: MockEmbeddingModel,
    index_path: String,
    metadata_path: String,
    vectors: RwLock<Vec<VectorEntry>>,
    metadata: RwLock<Vec<Review>>,

    config: IndexConfig,
    stats: RwLock<SearchStats>,
    query_cache: RwLock<HashMap<u64, CacheEntry>>,
}

impl VectorStore {
    pub async fn new(data_dir: &str) -> Result<Self> {

        std::fs::create_dir_all(data_dir)?;
        
        let index_path = format!("{}/reviews.index", data_dir);
        let metadata_path = format!("{}/reviews.jsonl", data_dir);
        

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


        store.load_from_disk().await?;
        
        Ok(store)
    }


    pub fn create_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(anyhow!("Cannot create embedding from empty text"));
        }
        Ok(self.embedding_model.embed(text))
    }

    pub async fn add_review(&self, review: Review) -> Result<()> {

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

        {
            let mut vectors = self.vectors.write().await;
            let mut metadata = self.metadata.write().await;
            
            vectors.push(vector_entry.clone());
            metadata.push(review.clone());
        }

        self.append_to_disk(&vector_entry, &review).await?;
        
        Ok(())
    }

    pub async fn add_reviews_bulk(&self, reviews: Vec<Review>) -> Result<usize> {
        let mut added_count = 0;
        
        for review in reviews {
            if let Ok(_) = self.add_review(review).await {
                added_count += 1;
            }
        }
        
        Ok(added_count)
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let start_time = Instant::now();

        let query_hash = self.hash_query(query);
        

        if let Some(cached_results) = self.check_cache(query_hash).await {
            self.update_stats_cache_hit().await;
            return Ok(cached_results.into_iter().take(limit).collect());
        }

        let query_vector = self.create_embedding(query)?;
        let query_array = Array1::from_vec(query_vector);
        

        let internal_limit = self.config.internal_result_num.max(limit * 2);
        
        let vectors = self.vectors.read().await;
        let metadata = self.metadata.read().await;
        
        let mut results = Vec::new();
        let mut vectors_compared = 0;
        

        for (i, vector_entry) in vectors.iter().enumerate() {
            let vector_array = Array1::from_vec(vector_entry.vector.clone());
            

            let combined_similarity = cosine_similarity(&query_array, &vector_array);
            
            vectors_compared += 1;
            

            if let Some(review) = metadata.get(i) {
                let mut final_similarity = combined_similarity;
                
  
                let query_lower = query.to_lowercase();
                let title_lower = review.review_title.to_lowercase();
                let body_lower = review.review_body.to_lowercase();
                
    
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
                

                if title_matches > 0 {
                    final_similarity += 0.15 * (title_matches as f32 / query_words.len() as f32); // Title match boost
                }
                
                if body_matches > 0 {
                    final_similarity += 0.1 * (body_matches as f32 / query_words.len() as f32); // Body match boost
                }
                
           
                if final_similarity >= self.config.similarity_threshold {
                    results.push(SearchResult {
                        review: review.clone(),
                        similarity_score: final_similarity,
                    });
                }
            }
        }
        
 
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        results.truncate(internal_limit);
        
    
        self.cache_results(query_hash, results.clone()).await;
        
   
        self.update_stats(start_time.elapsed().as_millis() as usize, vectors_compared).await;
        
    
        Ok(results.into_iter().take(limit).collect())
    }

 
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
       
            if entry.timestamp.elapsed().as_secs() < 300 {
                return Some(entry.results.clone());
            }
        }
        None
    }

    async fn cache_results(&self, query_hash: u64, results: Vec<SearchResult>) {
        let mut cache = self.query_cache.write().await;
        

        if cache.len() >= self.config.max_cache_size {

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


    pub async fn get_stats(&self) -> SearchStats {
        self.stats.read().await.clone()
    }


    async fn load_from_disk(&self) -> Result<()> {
    
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


    async fn append_to_disk(&self, vector_entry: &VectorEntry, review: &Review) -> Result<()> {
  
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.index_path)?;
            
            let vector_json = serde_json::to_string(vector_entry)?;
            writeln!(file, "{}", vector_json)?;
        }

   
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


    pub async fn count(&self) -> usize {
        let metadata = self.metadata.read().await;
        metadata.len()
    }


    

    pub fn get_config(&self) -> &IndexConfig {
        &self.config
    }
    

    pub async fn update_config(&mut self, new_config: IndexConfig) {
        self.config = new_config;
 
        let mut cache = self.query_cache.write().await;
        cache.clear();
    }
    

    pub async fn compact_cache(&self) {
        let mut cache = self.query_cache.write().await;
        let now = Instant::now();
        cache.retain(|_, entry| now.duration_since(entry.timestamp).as_secs() < 300);
    }
    

    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.query_cache.read().await;
        let stats = self.stats.read().await;
        (cache.len(), stats.cache_hits)
    }
    

    pub async fn add_reviews_batch(&self, reviews: Vec<Review>) -> Result<usize> {
        let mut added_count = 0;
        
        for review in reviews {
            if let Ok(()) = self.add_review(review).await {
                added_count += 1;
            }
        }
        

        let mut cache = self.query_cache.write().await;
        cache.clear();
        
        Ok(added_count)
    }
    

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
