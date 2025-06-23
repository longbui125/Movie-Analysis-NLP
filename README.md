# Movie Analysis NLP

A Python project applying Natural Language Processing (NLP) techniques to movie-related text data. This project enables exploration and analysis of plots, dialogues, reviews, and subtitles to extract meaningful insights, including theme classification, character networks, and interactive chatbot that mimics the character in the movie.

---

## Project Name

**Movie Analysis NLP**

---

## Purpose

This project focuses on leveraging NLP methods to analyze movie datasets, offering automated tools to:

- Classify content into thematic categories  
- Explore relationships between characters through network analysis  
- Interactively describe characters using a chatbot  
- Extract keywords, word frequencies, and visualize linguistic patterns  

Ideal for researchers, data scientists, or developers interested in combining NLP with entertainment data for exploratory analysis, educational projects, or building content-driven applications.

---

## Tech Stack

The project utilizes Python and popular open-source libraries:

- **Python** ≥ 3.8  
- **NLTK** – Basic text preprocessing (tokenization, stopword removal)  
- **spaCy** – Advanced NLP pipeline (lemmatization, named entity recognition)  
- **scikit-learn** – Theme classification and vectorization tasks  
- **pandas** – Data manipulation and dataset management  
- **matplotlib** & **seaborn** – Visualizations (charts, graphs)  
- **networkx** – Building and visualizing character networks  
- **wordcloud** – Word cloud visualizations  
- **Gradio** – Lightweight chatbot interface for character descriptions  

---

## Key Features

- **Data Ingestion**  
  Load and process movie-related text data from `.csv`, `.json`, or `.txt` files, including plot summaries, reviews, dialogues, or subtitles.

- **Text Preprocessing**  
  Clean and normalize text with tokenization, stop-word removal, and optional lemmatization or stemming.

- **Theme Classification**  
  Automatically classify dialogues or movie content into predefined themes (e.g., romance, conflict, comedy) using machine learning techniques.

- **Character Network Extraction**  
  Build interactive network graphs to visualize relationships and interactions between characters based on co-occurrence in dialogues or scenes.

- **Character Description Chatbot**  
  An interactive chatbot capable of providing character introductions or descriptions based on available movie data.

- **Keyword & Frequency Analysis**  
  Extract keywords, calculate word frequencies, and analyze the most relevant linguistic elements.

- **Visualization Tools**  
  Generate intuitive visualizations such as word clouds, bar charts, and interactive character networks to explore and present your analysis results.

---

## Getting Started

### Installation

```bash
git clone https://github.com/longbui125/Movie-Analysis-NLP.git
cd Movie-Analysis-NLP
pip install -r requirements.txt
cd Movie-Analysis-NLP; echo "huggingface_token=(your hugging face token here, place it in a .env file)"
cd Movie-Analysis-NLP && python gradio_app.py
