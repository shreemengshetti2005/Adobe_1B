
#  Adobe Hackathon 2025 1B

##  Overview

An advanced AI-powered PDF processing system designed to transform complex documents into actionable insights for analysts, researchers, and decision-makers. This pipeline automates the extraction, understanding, and ranking of information from PDFs, providing a robust solution for data-driven decision-making.

---

##  Key Features

###  PDF Parsing
Our robust PDF parsing engine accurately deconstructs PDF documents, identifying and extracting critical elements.
- **Semantic Structure Extraction:** Intelligently identifies and preserves the logical hierarchy of the document (headings, paragraphs, lists, etc.).
- **Table of Contents and Heuristic Analysis:** Leverages ToC for navigation and employs heuristic rules to infer document structure where explicit markers are absent.
- **Detailed Parsing Logs:** Provides comprehensive logs for transparency and debugging, showcasing the parsing process and any anomalies.
- **Structured JSON Output:** Exports parsed content into a well-defined JSON format, making it easily consumable for downstream applications.

###  Intelligent Embedding
Transforms parsed content into rich, context-aware vector embeddings, optimized for various analytical tasks.
- **Context-Aware Vector Embeddings:** Generates embeddings that capture the nuanced meaning and relationships within the text.
- **Persona and Goal-Driven Processing:** Allows for customization of embedding strategies based on the specific persona (e.g., legal analyst, medical researcher) and analytical goals, ensuring highly relevant representations.
- **Flexible Model Support:** Supports integration with a variety of embedding models, enabling adaptability to evolving AI landscapes and specific project requirements.

###  Smart Ranking
Our intelligent ranking system prioritizes information based on semantic relevance and customizable criteria.
- **Semantic Relevance Scoring:** Ranks document sections and snippets based on their conceptual similarity to user queries or predefined topics.
- **Optional Local LLM Enhancement:** Integrates with local Large Language Models (LLMs) to refine ranking based on deeper contextual understanding and inference.
- **Efficient Vector Search:** Utilizes highly optimized vector search algorithms for rapid retrieval of relevant information from large document corpuses.

###  Precise Snippet Extraction
Delivers highly targeted and relevant information snippets, minimizing noise and maximizing content utility.
- **Contextual Sentence Selection:** Extracts individual sentences or short passages that are most relevant to a query, while preserving their original context.
- **Diversity-Aware Retrieval:** Ensures a diverse range of relevant snippets are retrieved, preventing over-focus on a single perspective.
- **High-Relevance Content Focus:** Prioritizes snippets that contain the most critical information, directly addressing the user's need.

---

### Model Requirement for Docker Build

When building the Docker image, make sure the required model file is present in the correct location:
models/

This model is necessary for the application to function. We use the **Gemma-1B.Q4_K_M.gguf** model — a 1B parameter, 4-bit quantized GGUF version of Google's Gemma LLM.

####  To include it in your Docker image:

1. Place the GGUF model file inside the `models/` directory.
2. Modify your `Dockerfile` to include the following line:

```dockerfile
COPY models/Gemma-1B.Q4_K_M.gguf /app/models/Gemma-1B.Q4_K_M.gguf
```


##  Quick Start

Follow these steps to get the PDF Intelligence Pipeline up and running on your system.

### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.10+**: Essential for running the pipeline's core components.
- **Docker (optional)**: Recommended for a consistent and isolated environment.

### Installation

#### Python Dependencies
Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```
### OR

### Pull the docker image
```bash
docker pull voidisnull/asterix-adobe-1b:v3
```
#### And
```bash
docker run --rm -v $(pwd)/Challenge_1b:/app/Challenge_1b voidisnull/asterix-adobe-1b:v3
```
