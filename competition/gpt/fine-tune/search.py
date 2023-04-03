from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
import os
import multiprocessing as mp
from tqdm import tqdm
import pickle

cc_news = load_dataset('cc_news')
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

MAX_SEQ_LEN = 512

def article_to_sentences(article):
    article = article.replace('\n', ' ')
    sentences = []
    for sentence in article.split('. '):
        if len(sentence) > MAX_SEQ_LEN:
            sentences.append(sentence[:MAX_SEQ_LEN])
        if len(sentence) > 2:
            sentences.append(sentence+'.')
    return sentences


def search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, cc_news_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    result = []

    for score, idx in zip(top_results[0], top_results[1]):
        result.append({
            'text': cc_news['train']['text'][idx],
            'score': score.item()
        })

    return result


def main():
    # Search for the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 3
    for query in ['The first human to orbit the Earth was Yuri Gagarin.', 'The first human to orbit the Earth was John Glenn.']:
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, cc_news_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("Query:", query)
        print("Top 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(cc_news['train']['text'][idx], "(Score: %.4f)" % (score))

        print("\n\n")


if os.path.exists('cc_news_embeddings.pkl'):
    # If embeddings are already saved, load them
    with open('cc_news_embeddings.pkl', 'rb') as fIn:
        data = pickle.load(fIn)
        sentences = data['sentences']
        cc_news_embeddings = data['embeddings']
else:
    # Otherwise, generate them and save them
    # Break up each article into sentences, use multiprocessing to speed up the process
    # Check if sentences.pkl exists, if so, load it
    if os.path.exists('sentences.pkl'):
        with open('sentences.pkl', 'rb') as f:
            sentences = pickle.load(f)
        print("Loaded sentences from disk")
    else:
        print("Generating sentences using "+str(mp.cpu_count())+" cores...")
        sentences = []
        with mp.Pool(mp.cpu_count()) as pool:
            for article in tqdm(pool.imap_unordered(article_to_sentences, cc_news['train']['text']), total=len(cc_news['train']['text'])):
                sentences.append(article)
        # save sentences to pkl
        with open('sentences.pkl', 'wb') as f:
            pickle.dump(sentences, f)
        print("Done, generated %d sentences" % len(sentences))
    print("Generating embeddings...")
    cc_news_embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True, device='cuda')
    with open('cc_news_embeddings.pkl', 'wb') as f:
        pickle.dump(cc_news_embeddings, f)
    print("Done, saved embeddings to disk")
    # print("Normalizing embeddings...")
    # corpus_embeddings = cc_news_embeddings.to('cuda')
    # cc_news_embeddings_norm = util.normalize_embeddings(corpus_embeddings)
    # torch.save(cc_news_embeddings_norm, 'cc_news_embeddings_norm.pt')
    # print("Done, saved normalized embeddings to disk")


if __name__ == '__main__':
    main()