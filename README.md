# Article Search
Retreives similar articles to the one searched by using cosine similarity.

### Data Source
[CurationCorp News Articles Data](https://github.com/CurationCorp/curation-corpus)

### Downloading and Unpacking Word Embeddings
<b>Google News Vectors</b><br>
* Total Vocab: 3,000,000,000<br>
* Dimensions: 300
```bash
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gzip GoogleNews-vectors-negative300.bin.gz
```

<b>Output</b><br>
![](https://github.com/sidthakur08/article_search/blob/main/ss_ouput.jpg)


### Addition to this
1. developing an interface where you can search for a particular topic
2. use of fasttext to tackle out of vocabulary tokens
3. use of better similarity algorithms?
