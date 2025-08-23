**STKGNN: Scalable Spatio-Temporal Knowledge Graph Reasoning for Activity Recognition**


# README: Instructions Creating and Modelling Spatio-Temporal Knowledge Graphs (STKGs)
This repositry provides tools and scripts for creating and modeling custom **Spatio-Temporal Knowledge Graphs (STKGs)** from video datasets. The framework is adaptble and allow the users to modify parameters such as the number of videos, frames, or datasets which is essential to drive STKGs.

---
## **Creating Your Own STKGs**
1. Use the **`Create_STKGS`** Notebook.
2. Adjust parameters, such as:
   - Number of videos
   - Number of frames per video
   - Choice of video dataset
3. Customize the KG creation process to fit your dataset and application needs.

---
## **Use Pre-Created Knowledge Graphs in your implementations**
1. Use the **`ImportLocalData`** Python script to load one of the pre-generated KGs. 
   - All STKG data is in **STKG_LocalData** file
   - Each sub-file contains four `.txt` files for node_labels, node_features, edge, edge_features.
   
2. **Preprocess the KG Data**
   - Handle class imbalance issue by running the **`BalanceClassDistribution`** script.
   - This preprocsing step is recommended before applying models to raw STKGs.
   
---
## **Modelling STKGs**
We provide 3 models to analze and reason over the STKGs for activity understanding:
1. **`StableGCN.ipynb`**
   A lightweight model with general feature to bring stabilization of between temporal changes in STKGs.
   
2. **`TemporalSAGE.ipynb`**
   It has temporal support with medium-level model and provides specalized augmentation methods.
   
3. **`FusionGAT.ipynb`**
   Levrages Graph Autoencoders (GAE) recontruction approach and GAT with temporal support higher-level model for improved reasoning.

All the details about the models are shared in the paper.


---
### **How to Run the Models**
- Select a model script (`StableGCN`, `TemporalSAGE`, or `FusionGAT`) based on your needs and sclae of yoru STKGs 
(Note: The commands to import STKGs and preprocess the data are already embedded into models' scprits.)
- Adjust parameters such as model depth, learning rate, or batch size to fit your specific dataset or application.
- Each model contains both common and specialized implementations, offering flexibility for customization.

---
## **Comparative Analysis**
 Some pixel-wise state-of-the-art methods from the literature in our domain (the direct comparable works with our are not feasible, so we culd only implement limited similiar works) namely,  **`STIP-GCN_top1-5`** [1], **`AKU_top1-5`** [2], and **`TRG_top1-5`** [3] notebooks that have been implemented to be compatible with our dataset under **Comparative_Analysis** folder. Each has been reported by using Top-1 and Top-5 accuracy metrics and compared against our proposed models with corresponding notebooks are provided for evaluation.

[1] Sravani Yenduri, Vishnu Chalavadi, and C Krishna Mohan. 2022. STIP-GCN:Space-time interest points graph convolutional network for action recognition. In 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, US, 1–8.

[2] Yue Ma, Yali Wang, Yue Wu, Ziyu Lyu, Siran Chen, Xiu Li, and Yu Qiao. 2022. Visual knowledge graph for human action reasoning in videos. In Proc. 30th ACM International Conference on Multimedia. Association for Computing Machinery, New York, NY, USA, 4132–4141.

[3] Jingran Zhang, Fumin Shen, Xing Xu, and Heng Tao Shen. 2020. Temporal reasoning graph for activity recognition. IEEE Transactions on Image Processing 29 (2020), 5491–5506.

- Each model has also been adapted for "Top-1 and Top-5 accuracy" with following notebooks under **Evaluation_Top1-5** folder, **`FusionGAT_top1-5`**, **`StableGCN_top1-5`** and **`TemporalSAGE_top1-5`** computaton which are compatible for comparative analysis which are provided for direct execution by excluding reasoning parts of our models.

---
## **Environment Setup**
To replicate the environment of this repository:
- Use the provided **`Conda_spec-file.txt`** to create an identical Conda environment:
   ```bash
   conda create --name myenv --file Conda_spec-file.txt
   ```

---
## **Key Notes**
- The scripts are adaptable and modular and allow customization for various datasets and application daomains.
- The provided framework is scalable and designed for tasks requiring reasoning over spatio-temporal knowledge graph.

Thank you!
