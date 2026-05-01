# Learning from Disagreement: Multilingual Sexism Detection

## Overview

This project investigates sexism detection in social media under a **learning from disagreement** paradigm. Instead of relying on aggregated labels, the approach models **annotator-level disagreement** by training on **probability distributions (soft labels)**.

The task is inherently subjective, as different annotators may interpret the same content differently. This work aims to better capture such subjectivity and improve model performance by explicitly incorporating disagreement into the learning process.

---

## Objectives

- Develop models that predict **label distributions** rather than single labels  
- Compare **soft-label learning** with traditional **hard-label approaches**  
- Build a **multilingual system** (English and Spanish)  
- Evaluate the impact of disagreement modeling on performance  

---

## Tasks

The system addresses three classification levels:
 
1. Binary sexism detection  
2. Intention classification  
3. Fine-grained sexism categorization  

---

## Methodology

- Preprocess multilingual data and extract **soft labels** from annotators  
- Train transformer-based models (e.g., mBERT, XLM-RoBERTa)  
- Extend models to learn from **soft targets**  
- Design an **ensemble model** by aggregating predictions  
- Evaluate using both **standard metrics** and **distribution-based metrics**  

---

## Project Structure
``` 
sexism_project/
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── engine/
│   ├── utils/
│   └── losses/
│
├── data/
├── outputs/
├── notebooks/
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Expected Outcomes

- Implementation of baseline and soft-label models  
- Comparative analysis between hard and soft learning approaches  
- Evaluation of ensemble performance  
- Insights into modeling subjectivity in NLP tasks  

---