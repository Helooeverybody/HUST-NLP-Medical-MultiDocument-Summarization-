# Abstractive Long Multi-Document Summarization
Data is available at: [data](https://github.com/allenai/mslr-shared-task)

To run the code, please download the checkpoint files at, and put it folder model,and run 
notebook inference_test_output


Group 1 members:
- Nguyen Nhat Minh 20225510
- Doi Sy Thang - 20225528
- Ngo Duy Dat - 20225480
- Nguyen Hoang Son Tung - 20225536
- Vu Tuan Truong - 20225535

## Project Description

In this project, we try to investigate and learn both theoretical foundations and practical
approaches to solve the abstractive multi-document text summarization problem in natural lan-
guage processing(NLP), which remained highly challenging within linguistics and NLP be-
fore the emergence of transformers or attention-based large language models(LLMs) trained
on massive corpora. We propose to employ Longformer variant of BART, as well as further
Longformer-based architectures such as CENTRUM and PRIMERA. Furthermore, we adapt
Pegasus-X, a model originally developed for single-document summarization, to the multidoc-
ument setting. Furthermore, we experiment these models on Cochrane-medical dataset and
assess the performance based on the the ROUGE family of metrics, which are well established
for summarization tasks in the old days, and we further evaluate performance with BERTScore,
a more recent metric that measures not only tokens overlap but also the coherence of the gen-
erated summaries.


