SkinScan AI — Dermatology Prediction + RAG-Based Explanation System
===================================================================

🚀 Overview
-----------

**SkinScan AI** is a full-stack AI-powered dermatology analysis system that performs:

*   📸 Skin lesion classification (acne, melanoma, nevus, etc.)
    
*   🔍 Visual explainability using heatmaps (Grad-CAM style)
    
*   🧠 Retrieval-Augmented Generation (RAG) for medical explanations
    
*   💬 Context-aware Q&A about detected skin conditions
    

This project combines **Computer Vision + NLP + Full Stack Engineering** into a production-style system.

🧩 Problem Statement
--------------------

Most dermatology AI tools:

*   Only give predictions (no explanation)
    
*   Lack user interaction
    
*   Are not robust to low-quality images
    

👉 This system solves that by combining:

*   ML prediction
    
*   Explainability
    
*   Knowledge retrieval (RAG)
    
*   Interactive Q&A
    

🏗️ System Architecture
-----------------------

`   Frontend (React)      ↓  Node.js Backend (Express API)      ↓  ML Service (FastAPI)      ↓  [CV Model + RAG Engine]   `

🛠️ Tech Stack
--------------

### 💻 Frontend

*   React.js
    
*   Axios
    
*   Tailwind CSS
    
*   Component-based UI (Upload, Result, Q&A)
    

### 🌐 Backend (API Layer)

*   Node.js (Express)
    
*   Multer (file uploads)
    
*   Axios (service communication)
    

### 🤖 ML Service

*   FastAPI
    
*   PyTorch (ResNet-based classifier)
    
*   OpenCV (image processing)
    
*   NumPy
    

### 🧠 RAG Engine

*   Sentence Transformers (all-MiniLM-L6-v2)
    
*   FAISS (vector search)
    
*   Custom dermatology knowledge base
    

⚙️ ML Pipeline
--------------

### 1\. Data Collection

*   HAM10000 dataset (skin lesion dataset)
    
*   Organized into:
    
`   data/    train/      acne/      melanoma/      nevus/    val/   `

### 2\. Preprocessing

*   Resize images (224x224)
    
*   Normalize pixel values
    
*   Apply low-quality enhancement:
    
    *   Contrast enhancement (CLAHE)
        
    *   Noise reduction
        
    *   Sharpness correction
        

### 3\. Model Training

*   Base: ResNet18 (transfer learning)
    
*   Modified final layer for custom classes
    
`   model = models.resnet18(pretrained=True)  model.fc = nn.Linear(model.fc.in_features, num_classes)   `

*   Loss: CrossEntropyLoss
    
*   Optimizer: Adam
    

### 4\. Inference

*   Image → preprocess → model prediction
    
*   Output:
    
    *   Class label
        
    *   Confidence score
        

### 5\. Explainability (Heatmap)

*   Grad-CAM style visualization
    
*   Highlights regions influencing prediction
    

🧠 RAG Pipeline
---------------

### Step 1: Knowledge Base

Stored in:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   rag/data/derma_knowledge.txt   `

### Step 2: Embedding

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   model = SentenceTransformer("all-MiniLM-L6-v2")  embeddings = model.encode(docs)   `

### Step 3: Vector Index

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   index = faiss.IndexFlatL2(dim)  index.add(embeddings)   `

### Step 4: Retrieval

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   def retrieve(query):      q_emb = model.encode([query])      _, indices = index.search(q_emb, k)      return [docs[i] for i in indices[0]]   `

### Step 5: Answer Generation

*   Combines:
    
    *   Predicted condition
        
    *   Retrieved knowledge
        
    *   User question
        

🔄 How the App Works
--------------------

### 1\. User uploads image

→ React sends file to Node backend

### 2\. Node backend

→ forwards request to ML service

### 3\. ML service:

*   preprocesses image
    
*   predicts condition
    
*   generates heatmap
    
*   fetches explanation via RAG
    

### 4\. Response returned:

` {    "prediction": "acne",    "confidence": 0.98,    "explanation": "...",    "enhanced_image": "...",    "heatmap": "..."  }   `

### 5\. User asks question

→ frontend sends to /ask

→ RAG retrieves knowledge + generates answer

📦 Project Structure
--------------------

`   derma-ai-project/  │  ├── frontend/  │   ├── components/  │   │   ├── Upload.js  │   │   └── Result.js  │  ├── backend/  │   ├── server.js  │  ├── ml-service/  │   ├── api/  │   │   └── main.py  │   ├── preprocessing/  │   ├── inference/  │   ├── training/  │   ├── rag/  │   └── models/   `

⚡ Setup Instructions
--------------------

### 1\. Clone repo
`   git clone https://github.com/your-username/derma-ai.git  cd derma-ai   `

### 2\. Start ML Service
`   cd ml-service  python -m venv venv  source venv/bin/activate  pip install -r requirements.txt  uvicorn api.main:app --reload --port 8001   `

### 3\. Start Backend
`   cd backend  npm install  node server.js   `

### 4\. Start Frontend

`   cd frontend  npm install  npm run dev   `

🌐 API Endpoints
----------------

### Predict
`   POST /predict   `

Form-data:

`   file: image   `

### Ask (RAG Q&A)
`   POST /ask   `

Body:

`   {    "prediction": "acne",    "question": "Is this dangerous?"  }   `

🚀 Features
-----------

*   ✅ Skin disease classification
    
*   ✅ Explainable AI (heatmaps)
    
*   ✅ RAG-based explanation
    
*   ✅ Interactive Q&A system
    
*   ✅ Low-quality image enhancement
    
*   ✅ Full-stack integration
    

⚠️ Limitations
--------------

*   Not a medical diagnostic tool
    
*   Depends on dataset quality
    
*   Limited knowledge base for RAG
    

🔮 Future Improvements
----------------------

*   Mobile deployment
    
*   Larger dermatology dataset
    
*   LLM integration for richer answers
    
*   Real-time camera input
    
*   Multi-condition detection
    

📌 Conclusion
-------------

SkinScan AI demonstrates how **Computer Vision + RAG + Full Stack Systems** can be combined into a real-world AI product.

This project goes beyond simple ML models by delivering:

*   Explainability
    
*   User interaction
    
*   Scalable architecture
    

👨‍💻 Author
------------

Built as part of a production-style AI system project.
