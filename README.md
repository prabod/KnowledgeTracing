# Knowledge Tracing Model Implementation

## Objectives
This project implements and evaluates a Deep Knowledge Tracing (DKT) model with two variants:
1. Standard DKT model with configurable question embeddings and previous response features
2. DKT model with interaction features between concepts and responses

The primary objective is to predict student performance on future questions based on their historical interaction data, enabling personalized learning recommendations and adaptive learning systems.

## Methodology Details

### Model Architecture
The implementation includes two main model variants:

1. **Standard DKT Model**
   - LSTM-based architecture for sequential learning
   - Configurable input features:
     - Question embeddings (optional)
     - Previous response features (optional)
   - Binary cross-entropy loss function
   - Adam optimizer with learning rate scheduling

2. **DKT Model with Interaction**
   - Enhanced architecture incorporating interaction features
   - Embedding layer for question-response interactions
   - LSTM layer for sequential learning
   - Output layer for prediction
   - Binary cross-entropy loss function
   - Adam optimizer with learning rate scheduling

### Training Process
- Batch size: 64
- Maximum sequence length: 200
- Learning rate: 0.01
- Maximum epochs: 20
- Validation fold: 4
- Early stopping based on validation loss
- Learning rate reduction on plateau
- Deterministic training with fixed seeds for reproducibility

## Exploratory Data Analysis

### Dataset Overview
The XES3G5M dataset consists of:
- 33,397 interaction sequences
- 14,453 unique users
- 865 unique concepts
- 5-fold cross-validation split

### Key Statistics
1. **User Interaction Patterns**
   - Average sessions per user: 2.31
   - Average questions per session: 153.88
   - Median questions per session: 200
   - Average time between questions: 20.06 minutes

2. **Question Distribution**
   - Total unique questions: 7,619
   - Maximum attempts per question: 23,488
   - Minimum attempts per question: 1
   - Mean attempts per question: 674.59
   - Median attempts per question: 193

3. **Concept Distribution**
   - Total unique concepts: 865
   - Most frequent concept: 120,791 attempts
   - Concept attempts follow a power-law distribution
   - Strong correlation between concept frequency and question difficulty

4. **Response Patterns**
   - Average correct response rate: 79.50%
   - Standard deviation: 14.20%
   - Median correct response rate: 82.50%
   - Wide variation in question difficulty (0-100% correct)

5. **Temporal Analysis**
   - Average time between questions: 20.06 minutes
   - Standard deviation: 11.60 minutes
   - Median time: 18.04 minutes
   - Maximum time: 262.97 minutes

### Data Characteristics
1. **Sequence Length**
   - Most sequences are padded to 200 interactions
   - Shorter sequences show normal distribution
   - Average length of non-padded sequences: 92.56

2. **Question Embeddings**
   - 768-dimensional embeddings
   - T-SNE visualization shows clustering patterns
   - Embeddings capture semantic relationships

3. **Concept Relationships**
   - Strong correlation between related concepts

## Result Details

### Training Configuration
- Learning rate: 0.01
- Hidden dimension: 100
- Input dimension: 768
- Number of runs: 10 (for statistical significance)

### Evaluation Results

| Model | Test Loss | Test AUC | Question Embeddings | Previous Responses | Interaction Features |
|-------|-----------|----------|---------------------|-------------------|---------------------|
| Standard DKT | 0.4287 ± 0.0012 | 0.7817 ± 0.0010 | Disabled | Enabled | Disabled |
| DKT with Interaction | 0.3652 ± 0.0004 | 0.7668 ± 0.0004 | Disabled| Enabled | Enabled |

## How to Run

1. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/prabod/KnowledgeTracing.git
   cd knowledge-tracing

   # Install dependencies
   make uv
   ```

2. **Training the Standard DKT Model**
   ```bash
   python train.py \
     --use-question-embeddings False \
     --use-previous-responses True \
     --input-dim 768 \
     --hidden-dim 100 \
     --lr 0.01 \
     --batch-size 64 \
     --max-seq-length 200 \
     --val-fold 4 \
     --max-epochs 20 \
     --repeat 10

   # or use make command to use default parameters
   make train
   ```

3. **Training the DKT Model with Interaction**
   ```bash
   python train.py \
     --with-interaction True \
     --number-of-concepts 865 \
     --embed-size 100 \
     --hidden-dim 100 \
     --lr 0.01 \
     --batch-size 64 \
     --max-seq-length 200 \
     --val-fold 4 \
     --max-epochs 20 \
     --repeat 10

   # or use make command to use default parameters
   make train-with-interaction
   ```

## Concluding Remarks

### Strengths
   - Simple and interpretable architecture
   - Efficient training with fewer parameters
   - Better modeling of concept interactions
   - Enhanced feature representation through embeddings
   - Potential for improved prediction accuracy

### Weaknesses
   - Limited ability to capture complex interactions
   - May miss important patterns in question-response sequences
   - Less effective with sparse interaction data
   - Higher computational complexity
   - Slower due to sequential nature of RNNs

### Opportunities for Future Work
1. **Enhanced Features**:
   - Incorporate temporal features
   - Add student demographic information
   - Include question difficulty levels
   - Integrate concept hierarchies

2. **Architecture Improvements**:
   - Implement attention mechanisms (SAKT etc.)
   - Add transformer-based components
   - Investigate ensemble methods

3. **Data Enrichment**:
   - Collect more diverse student interaction data
   - Include detailed question metadata
   - Add concept mastery progression data
   - Incorporate student feedback and engagement metrics

4. **Deployment Considerations**:
   - Real-time prediction capabilities
   - Model interpretability features
   - Continuous learning mechanisms 