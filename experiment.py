import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSESFramework:
    """
    Simplified implementation of the MoSEs (Mixture of Stylistic Experts) framework
    for AI-generated text detection based on the research paper.
    """
    
    def __init__(self, n_prototypes=5, n_components=32, random_state=42):
        """
        Initialize the MoSEs framework components.
        
        Args:
            n_prototypes: Number of prototypes per style for SAR
            n_components: Number of PCA components for semantic feature compression
            random_state: Random seed for reproducibility
        """
        self.n_prototypes = n_prototypes
        self.n_components = n_components
        self.random_state = random_state
        
        # Core components
        self.srr_data = None  # Stylistics Reference Repository
        self.prototypes = {}  # Stylistics-Aware Router prototypes
        self.cte_model = None  # Conditional Threshold Estimator
        self.pca = None  # PCA for semantic feature compression
        self.scaler = None  # Scaler for feature normalization
        
        logger.info("MoSEs framework initialized with %d prototypes and %d PCA components", 
                   n_prototypes, n_components)
    
    def extract_linguistic_features(self, texts):
        """
        Extract linguistic features from texts as described in the paper.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of linguistic features for each text
        """
        logger.info("Extracting linguistic features from %d texts", len(texts))
        
        features = []
        for text in texts:
            # Simulate feature extraction (in a real implementation, these would be calculated)
            text_length = len(text.split())
            log_prob_mean = np.random.normal(0, 1)  # Simulated
            log_prob_var = np.random.normal(1, 0.5)  # Simulated
            ngram_repetition_2 = np.random.uniform(0, 0.5)  # Simulated
            ngram_repetition_3 = np.random.uniform(0, 0.3)  # Simulated
            type_token_ratio = np.random.uniform(0.2, 0.8)  # Simulated
            
            features.append([
                text_length, log_prob_mean, log_prob_var, 
                ngram_repetition_2, ngram_repetition_3, type_token_ratio
            ])
        
        return np.array(features)
    
    def extract_semantic_embeddings(self, texts):
        """
        Extract semantic embeddings from texts (simulated for this implementation).
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of semantic embeddings for each text
        """
        logger.info("Extracting semantic embeddings from %d texts", len(texts))
        
        # In a real implementation, this would use a pre-trained model like BGE-M3
        # For this simplified version, we generate random embeddings
        embeddings = np.random.randn(len(texts), 512)  # Simulated 512-dim embeddings
        
        # Apply PCA for dimensionality reduction
        if self.pca is None:
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            embeddings_reduced = self.pca.fit_transform(embeddings)
        else:
            embeddings_reduced = self.pca.transform(embeddings)
        
        return embeddings_reduced
    
    def build_srr(self, texts, labels, styles):
        """
        Build the Stylistics Reference Repository (SRR).
        
        Args:
            texts: List of text strings
            labels: List of labels (0 for human, 1 for AI-generated)
            styles: List of style categories for each text
        """
        logger.info("Building Stylistics Reference Repository with %d samples", len(texts))
        
        # Extract features
        linguistic_features = self.extract_linguistic_features(texts)
        semantic_embeddings = self.extract_semantic_embeddings(texts)
        
        # Combine features
        combined_features = np.hstack([linguistic_features, semantic_embeddings])
        
        # Normalize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            combined_features = self.scaler.fit_transform(combined_features)
        else:
            combined_features = self.scaler.transform(combined_features)
        
        # Store SRR data - convert lists to numpy arrays for proper indexing
        self.srr_data = {
            'texts': texts,
            'labels': np.array(labels),
            'styles': np.array(styles),  # Convert to numpy array
            'features': combined_features,
            'linguistic_features': linguistic_features,
            'semantic_embeddings': semantic_embeddings
        }
        
        logger.info("SRR built successfully with %d samples", len(texts))
    
    def train_sar(self):
        """
        Train the Stylistics-Aware Router (SAR) using prototype-based approximation.
        """
        if self.srr_data is None:
            logger.error("SRR not built. Please call build_srr() first.")
            sys.exit(1)
        
        logger.info("Training Stylistics-Aware Router with %d prototypes per style", self.n_prototypes)
        
        # Group data by style
        styles = np.unique(self.srr_data['styles'])
        self.prototypes = {}
        
        for style in styles:
            style_mask = self.srr_data['styles'] == style
            style_indices = np.where(style_mask)[0]
            style_embeddings = self.srr_data['semantic_embeddings'][style_mask]
            
            # Check if we have enough samples for the number of prototypes
            n_samples = len(style_embeddings)
            if n_samples == 0:
                logger.warning("No samples found for style: %s, skipping", style)
                continue
            
            # Adjust number of prototypes if necessary
            n_prototypes_for_style = min(self.n_prototypes, n_samples)
            
            if n_prototypes_for_style == 1:
                # If only one prototype, use the mean of all samples
                centroids = np.array([np.mean(style_embeddings, axis=0)])
            else:
                # Use K-means to find prototypes
                try:
                    kmeans = KMeans(
                        n_clusters=n_prototypes_for_style, 
                        random_state=self.random_state, 
                        n_init=10
                    )
                    kmeans.fit(style_embeddings)
                    centroids = kmeans.cluster_centers_
                except Exception as e:
                    logger.warning("KMeans failed for style %s with error: %s. Using mean instead.", style, str(e))
                    centroids = np.array([np.mean(style_embeddings, axis=0)])
            
            self.prototypes[style] = {
                'centroids': centroids,
                'indices': style_indices
            }
        
        logger.info("SAR trained successfully with prototypes for %d styles", len(self.prototypes))
    
    def sar_route(self, text_embedding):
        """
        Route input text to appropriate reference samples using SAR.
        
        Args:
            text_embedding: Semantic embedding of the input text
            
        Returns:
            Indices of activated reference samples
        """
        if not self.prototypes:
            logger.error("SAR not trained. Please call train_sar() first.")
            sys.exit(1)
        
        activated_indices = []
        
        # Ensure text_embedding is 1D
        if text_embedding.ndim > 1:
            text_embedding = text_embedding.flatten()
        
        # Find nearest prototypes across all styles
        for style, proto_data in self.prototypes.items():
            centroids = proto_data['centroids']
            
            # Calculate distances to all prototypes of this style
            # Ensure proper broadcasting
            if centroids.ndim == 1:
                centroids = centroids.reshape(1, -1)
            
            distances = np.linalg.norm(centroids - text_embedding.reshape(1, -1), axis=1)
            
            # Get indices of the nearest prototypes (use min to avoid index errors)
            n_prototypes_to_select = min(len(centroids), self.n_prototypes)
            nearest_proto_indices = np.argsort(distances)[:n_prototypes_to_select]
            
            # Add all samples from this style (simplified approach)
            activated_indices.extend(proto_data['indices'])
        
        # Remove duplicates and return
        return list(set(activated_indices))
    
    def train_cte(self):
        """
        Train the Conditional Threshold Estimator (CTE).
        """
        if self.srr_data is None:
            logger.error("SRR not built. Please call build_srr() first.")
            sys.exit(1)
        
        logger.info("Training Conditional Threshold Estimator")
        
        # Use all SRR data for training in this simplified version
        X = self.srr_data['features']
        y = self.srr_data['labels']
        
        # Train a classifier (simplified version of the paper's CTE)
        # In the paper, they use logistic regression or XGBoost
        try:
            self.cte_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=6
            )
            
            self.cte_model.fit(X, y)
            logger.info("CTE trained successfully with %d samples", len(y))
        except Exception as e:
            logger.error("Failed to train CTE: %s", str(e))
            sys.exit(1)
    
    def predict(self, texts, styles=None):
        """
        Predict whether texts are AI-generated using the MoSEs framework.
        
        Args:
            texts: List of text strings to classify
            styles: Optional list of style categories for each text
            
        Returns:
            predictions: List of predicted labels (0 for human, 1 for AI-generated)
            confidences: List of confidence scores
        """
        if self.cte_model is None:
            logger.error("CTE not trained. Please call train_cte() first.")
            sys.exit(1)
        
        logger.info("Predicting labels for %d texts", len(texts))
        
        try:
            # Extract features from input texts
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            
            # For each text, use SAR to find relevant reference samples
            all_activated_indices = []
            for i, embedding in enumerate(semantic_embeddings):
                activated_indices = self.sar_route(embedding)
                all_activated_indices.append(activated_indices)
            
            # Combine features and make predictions
            combined_features = np.hstack([linguistic_features, semantic_embeddings])
            combined_features = self.scaler.transform(combined_features)
            
            predictions = self.cte_model.predict(combined_features)
            confidences = np.max(self.cte_model.predict_proba(combined_features), axis=1)
            
            logger.info("Prediction completed: %d AI-generated, %d human texts detected", 
                       sum(predictions), len(predictions) - sum(predictions))
            
            return predictions, confidences
        
        except Exception as e:
            logger.error("Prediction failed: %s", str(e))
            sys.exit(1)

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        texts: List of synthetic text samples
        labels: List of labels (0 for human, 1 for AI-generated)
        styles: List of style categories
    """
    logger.info("Generating %d synthetic text samples", n_samples)
    
    texts = []
    labels = []
    styles = []
    
    # Define some style categories
    style_categories = ['news', 'academic', 'dialogue', 'story', 'comment']
    
    for i in range(n_samples):
        # Randomly assign label and style
        label = np.random.choice([0, 1])  # 0 = human, 1 = AI-generated
        style = np.random.choice(style_categories)
        
        # Generate synthetic text based on style and label
        if style == 'news':
            text = "This is a news article about recent developments in technology. " + \
                  "Experts are discussing the implications of these advances."
        elif style == 'academic':
            text = "In this paper, we present a novel approach to solving the problem. " + \
                  "Our method demonstrates significant improvements over existing techniques."
        elif style == 'dialogue':
            text = "Person A: How are you doing today? Person B: I'm doing well, thank you. " + \
                  "How about yourself?"
        elif style == 'story':
            text = "Once upon a time, in a land far away, there lived a brave knight. " + \
                  "The knight embarked on a quest to save the kingdom from danger."
        else:  # comment
            text = "I really enjoyed this content. It was informative and well-presented. " + \
                  "Looking forward to more like this!"
        
        # Add some variation based on label
        if label == 1:  # AI-generated
            text = text + " The content continues with additional details and explanations."
        
        texts.append(text)
        labels.append(label)
        styles.append(style)
    
    return texts, labels, styles

def main():
    """Main function to demonstrate the MoSEs framework."""
    logger.info("Starting MoSEs framework demonstration")
    
    try:
        # Generate synthetic data
        texts, labels, styles = generate_synthetic_data(n_samples=1000)
        
        # Split into reference and test sets (80/20)
        split_idx = int(0.8 * len(texts))
        ref_texts, test_texts = texts[:split_idx], texts[split_idx:]
        ref_labels, test_labels = labels[:split_idx], labels[split_idx:]
        ref_styles, test_styles = styles[:split_idx], styles[split_idx:]
        
        logger.info("Data split: %d reference samples, %d test samples", 
                   len(ref_texts), len(test_texts))
        
        # Initialize and train MoSEs framework
        meses = MoSESFramework(n_prototypes=3, n_components=16)
        
        # Build SRR
        meses.build_srr(ref_texts, ref_labels, ref_styles)
        
        # Train SAR
        meses.train_sar()
        
        # Train CTE
        meses.train_cte()
        
        # Make predictions on test set
        predictions, confidences = meses.predict(test_texts, test_styles)
        
        # Evaluate performance
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        logger.info("Evaluation results:")
        logger.info("Accuracy: %.4f", accuracy)
        logger.info("F1 Score: %.4f", f1)
        
        # Print detailed results
        test_labels_array = np.array(test_labels)
        human_correct = sum((predictions == 0) & (test_labels_array == 0))
        human_total = sum(test_labels_array == 0)
        ai_correct = sum((predictions == 1) & (test_labels_array == 1))
        ai_total = sum(test_labels_array == 1)
        
        logger.info("Human text detection: %d/%d (%.2f%%)", 
                   human_correct, human_total, 100 * human_correct/human_total if human_total > 0 else 0)
        logger.info("AI-generated text detection: %d/%d (%.2f%%)", 
                   ai_correct, ai_total, 100 * ai_correct/ai_total if ai_total > 0 else 0)
        
        # Summary of results
        logger.info("MoSEs framework demonstration completed successfully")
        logger.info("The simplified implementation shows the core components:")
        logger.info("1. SRR with linguistic and semantic features")
        logger.info("2. SAR with prototype-based routing")
        logger.info("3. CTE with conditional threshold estimation")
        logger.info("For real applications, replace synthetic features with actual linguistic analysis")
        logger.info("and use pre-trained embeddings from models like BGE-M3")
        
    except Exception as e:
        logger.error("Error in MoSEs framework: %s", str(e))
        logger.error("Terminating due to critical error")
        sys.exit(1)

if __name__ == "__main__":
    main()