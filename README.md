1. First obtain the word embedding matrix of English and French according to the paper.
2. Run make_en.py and make_fr.py.
3. Get lex.1.e2f and lex.1.f2e from CLSP Grid, and put them in the same directory.
4. Uncomment the comment line of create_language_pairs.py, run the code.
5. Comment the comment line if you would like to speed it up for further use.
6. Run bilingual.py, you can check the argument, and maybe the code position needs to be modified



The rest of them are for training word embeddings, and evaluation of the result of word embedding.

1. Get ant corpus.
2. Run tokenization by Moses.
3. Run remove.py, which removes the name and number, replaced by <name>, <num>.
4. Run window.py, which takes consective 5 words to one line.
5. If you think the corpus is too large, you can always run shrink.py, and specify any size.
6. Then run train.py, there are arguments to be specified as bilingual.py.
7. If you have done training, you can run evaluation.py to evaluate the result.