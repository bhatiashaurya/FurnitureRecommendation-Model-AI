# Model Training Notebook - Complete Guide

This notebook trains the following models for the furniture recommendation system:

## Models to Train:

1. **Text Embeddings (NLP)**
   - Sentence-BERT embeddings for semantic search
   - TF-IDF vectorizer for traditional similarity

2. **Image Features (CV)**
   - ResNet or ViT for image feature extraction
   - CNN-based product category classifier

3. **Recommendation System**
   - Content-based filtering
   - Hybrid model combining text + image

4. **Vector Database Integration**
   - Upload embeddings to Pinecone
   - Test semantic search

5. **GenAI Integration**
   - LangChain setup for description generation
   - Prompt engineering

## Notebook Structure:

### Part 1: Data Loading and Preprocessing
```python
import pandas as pd
import numpy as np
from backend.utils.preprocessor import DataPreprocessor

# Load data
df = pd.read_csv('../intern_data_ikarus.csv')

# Preprocess
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_dataframe(df)
```

### Part 2: Text Embeddings (NLP)
```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
text_embeddings = model.encode(df_processed['combined_text'].tolist())

# Save embeddings
np.save('backend/models/text_embeddings.npy', text_embeddings)
```

### Part 3: Image Feature Extraction (CV)
```python
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ResNet
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Feature extraction function
def extract_image_features(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        img_t = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            features = resnet(img_t)
        
        return features.numpy().flatten()
    except:
        return np.zeros(1000)  # Return zero vector on error

# Extract for all products
image_features = []
for images in df_processed['images_list']:
    if images:
        feat = extract_image_features(images[0])
        image_features.append(feat)
    else:
        image_features.append(np.zeros(1000))

image_features = np.array(image_features)
np.save('backend/models/image_features.npy', image_features)
```

### Part 4: Combined Embeddings
```python
# Normalize embeddings
from sklearn.preprocessing import StandardScaler

scaler_text = StandardScaler()
text_embeddings_norm = scaler_text.fit_transform(text_embeddings)

scaler_image = StandardScaler()
image_features_norm = scaler_image.fit_transform(image_features)

# Combine embeddings (weighted average)
alpha = 0.7  # Weight for text
beta = 0.3   # Weight for image

combined_embeddings = alpha * text_embeddings_norm + beta * image_features_norm
```

### Part 5: Upload to Pinecone
```python
from backend.utils.pinecone_client import PineconeClient

# Initialize Pinecone
pinecone_client = PineconeClient()
pinecone_client.create_index()

# Prepare metadata
metadata = []
for _, row in df_processed.iterrows():
    metadata.append({
        'uniq_id': row['uniq_id'],
        'title': row['title'],
        'brand': row['brand'],
        'price': float(row['price_numeric']) if not pd.isna(row['price_numeric']) else 0.0,
        'category': row['main_category'],
        'image': row['images_list'][0] if row['images_list'] else '',
        'description': row['description'] if not pd.isna(row['description']) else ''
    })

# Upload embeddings
pinecone_client.upsert_embeddings(combined_embeddings, metadata)
```

### Part 6: Test Recommendations
```python
# Test semantic search
query = "modern black leather dining chair"
query_embedding = model.encode([query])[0]

# Normalize query embedding
query_norm = scaler_text.transform([query_embedding])[0]

# Search Pinecone
results = pinecone_client.search(query_norm, top_k=5)

print("Top 5 Recommendations:")
for i, result in enumerate(results):
    print(f"\n{i+1}. {result['metadata']['title']}")
    print(f"   Brand: {result['metadata']['brand']}")
    print(f"   Score: {result['score']:.3f}")
```

### Part 7: GenAI Integration Test
```python
from backend.utils.genai_generator import ProductDescriptionGenerator

# Initialize generator
gen = ProductDescriptionGenerator()

# Generate descriptions for top results
for result in results[:3]:
    desc = gen.generate_description(result['metadata'])
    print(f"\nProduct: {result['metadata']['title']}")
    print(f"Generated Description: {desc}")
```

### Part 8: Save Models and Configurations
```python
import pickle

# Save text model configuration
model_config = {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'dimension': 384,
    'alpha': alpha,
    'beta': beta
}

with open('backend/models/config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

# Save scalers
with open('backend/models/scaler_text.pkl', 'wb') as f:
    pickle.dump(scaler_text, f)

with open('backend/models/scaler_image.pkl', 'wb') as f:
    pickle.dump(scaler_image, f)

print("✓ All models saved successfully!")
```

## Evaluation Metrics:

1. **Recommendation Quality**
   - Precision@K
   - Recall@K
   - NDCG (Normalized Discounted Cumulative Gain)

2. **Semantic Search Performance**
   - Average similarity score
   - Category match rate
   - Price range accuracy

3. **GenAI Quality**
   - Manual evaluation of descriptions
   - Relevance score
   - Creativity assessment

## Next Steps:

1. Run this notebook cell by cell
2. Monitor performance metrics
3. Fine-tune hyperparameters (alpha, beta)
4. Test with different embedding models
5. Evaluate GenAI outputs
6. Deploy to production

---

**Note**: This is a guide. Create the actual Jupyter notebook and implement each section with proper error handling and logging.
