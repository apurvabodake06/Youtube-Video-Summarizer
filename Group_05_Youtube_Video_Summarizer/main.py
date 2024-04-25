from youtube_transcript_api import YouTubeTranscriptApi
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import math
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None

def write_string_to_file(text, filename):
    try:
        with open(filename, 'w') as file:
            file.write(text)
        print("String successfully written to", filename)
    except Exception as e:
        print("Error:", e)

def create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix

# ... (other functions)
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

def _find_average_score(sentenceValue) -> float:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def main(url):
    video_id = url  # Replace with the actual video ID
    transcript = get_transcript(video_id) #kKvK2foOTJM?si=NtrzBBkliixRMIxa

    if transcript:
        transcript_text = " ".join(entry['text'] for entry in transcript)
        sentences = sent_tokenize(transcript_text)

        # Store subtitles in subtitle44.txt
        subtitle_filename = "subtitle44.txt"
        write_string_to_file(transcript_text, subtitle_filename)

        # 2 Create the Frequency matrix of the words in each sentence.
        freq_matrix = create_frequency_matrix(sentences)

        # 3 Calculate TermFrequency and generate a matrix
        tf_matrix = _create_tf_matrix(freq_matrix)

        # 4 creating table for documents per words
        count_doc_per_words = _create_documents_per_words(freq_matrix)

        # 5 Calculate IDF and generate a matrix
        idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, len(sentences))

        # 6 Calculate TF-IDF and generate a matrix
        tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)

        # 7 Important Algorithm: score the sentences
        sentence_scores = _score_sentences(tf_idf_matrix)

        # 8 Find the threshold
        threshold = _find_average_score(sentence_scores)

        # 9 Important Algorithm: Generate the summary
        summary = _generate_summary(sentences, sentence_scores, 0.9 * threshold)

        # Write the summary to output55.txt
        summary_filename = "output55.txt"
        write_string_to_file(summary, summary_filename)

    else:
        print("Transcript not available.")

#if __name__ == "__main__":
  #  main()

# transcript.py

def generate_summary_from_file(url):
    main(url)
    try:
        with open('output55.txt', 'r') as file:
            summary = file.read()
            return summary
    except Exception as e:
        print(f"Error reading summary from file: {e}")
        return None
