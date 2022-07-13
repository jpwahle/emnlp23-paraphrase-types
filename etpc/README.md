---
annotations_creators:
- crowdsourced
language_creators:
- found
language:
- en
license:
- unknown
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- sentiment-classification
pretty_name: Extended Paraphrase Typology Corpus
---
# Dataset Card for [Dataset Name]
## Table of Contents
- [Dataset Card for \[Dataset Name\]](#dataset-card-for-dataset-name)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)
## Dataset Description
- **Homepage:** https://github.com/venelink/ETPC/
- **Repository:**
- **Paper:** [ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation](http://www.lrec-conf.org/proceedings/lrec2018/pdf/661.pdf)
- **Leaderboard:**
- **Point of Contact:**
### Dataset Summary
We present the Extended Paraphrase Typology (EPT) and the Extended Typology Paraphrase Corpus (ETPC). The EPT typology addresses several practical limitations of existing paraphrase typologies: it is the first typology that copes with the non-paraphrase pairs in the paraphrase identification corpora and distinguishes between contextual and habitual paraphrase types. ETPC is the largest corpus to date annotated with atomic paraphrase types. It is the first corpus with detailed annotation of both the paraphrase and the non-paraphrase pairs and the first corpus annotated with paraphrase and negation. Both new resources contribute to better understanding the paraphrase phenomenon, and allow for studying the relationship between paraphrasing and negation. To the developers of Paraphrase Identification systems ETPC corpus offers better means for evaluation and error analysis. Furthermore, the EPT typology and ETPC corpus emphasize the relationship with other areas of NLP such as Semantic Similarity, Textual Entailment, Summarization and Simplification.
### Supported Tasks and Leaderboards
- `text-classification`
### Languages
The text in the dataset is in English (`en`).
## Dataset Structure
### Data Fields
- `idx`: Monotonically increasing index ID.
- `sentence1`: Complete sentence expressing an opinion about a film.
- `sentence2`: Complete sentence expressing an opinion about a film.
- `etpc_label`: Whether the text-pair is a paraphrase, either "yes" (1) or "no" (0) according to etpc annotation schema.
- `mrpc_label`: Whether the text-pair is a paraphrase, either "yes" (1) or "no" (0) according to mrpc annotation schema.
- `negation`: Whether on sentence is a negation of another, either "yes" (1) or "no" (0).
### Data Splits
train: 5801

## Dataset Creation
### Curation Rationale
[More Information Needed]
### Source Data
#### Initial Data Collection and Normalization
[More Information Needed]
#### Who are the source language producers?
Rotten Tomatoes reviewers.
### Annotations
#### Annotation process
[More Information Needed]
#### Who are the annotators?
[More Information Needed]
### Personal and Sensitive Information
[More Information Needed]
## Considerations for Using the Data
### Social Impact of Dataset
[More Information Needed]
### Discussion of Biases
[More Information Needed]
### Other Known Limitations
[More Information Needed]
## Additional Information
### Dataset Curators
[More Information Needed]
### Licensing Information
Unknown.
### Citation Information
```bibtex
@inproceedings{kovatchev-etal-2018-etpc,
    title = "{ETPC} - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation",
    author = "Kovatchev, Venelin  and
      Mart{\'\i}, M. Ant{\`o}nia  and
      Salam{\'o}, Maria",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1221",
}
```
### Contributions
Thanks to [@jpwahle](https://github.com/jpwahle) for adding this dataset.