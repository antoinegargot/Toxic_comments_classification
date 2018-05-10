library(tidyverse)
library(magrittr)
library(text2vec)
library(tokenizers)
library(xgboost)
library(glmnet)
library(doParallel)
library(tm)
library(caret)
registerDoParallel(4)

#Importing the data
data_set <- read.csv("DATA/train.csv")
targets <- c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
tri <- 1:nrow(train)

tr_te <- data_set %>% 
  select(-one_of(targets)) %>% 
  mutate(length = str_length(comment_text), ncap = str_count(comment_text, "[A-Z]"), ncap_len = ncap / length, nexcl = str_count(comment_text, fixed("!")), nquest = str_count(comment_text, fixed("?")), npunct = str_count(comment_text, "[[:punct:]]"), nword = str_count(comment_text, "\\w+"), nsymb = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^")) %>% 
  select(-id) %T>% 
  glimpse()

it <- tr_te %$%
  str_to_lower(comment_text) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  itoken(tokenizer = tokenize_word_stems)

vectorizer <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("en")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.5, vocab_term_max = 4000) %>%
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <- create_dtm(it, vectorizer) %>%
  fit_transform(m_tfidf)  

set.seed(42)
smp_size <- floor(0.80 * nrow(data_set))
train_ind <- sample(seq_len(nrow(data_set)), size = smp_size)
train <- data_set[train_ind, ]
test <- data_set[-train_ind, ]

X <- tr_te %>% 
  select(-comment_text) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(tfidf)

X_test <- X[-tri, ]
X <- X[tri, ]
result <- test[-1]
for (target in targets) {
  y <- train[[target]]
  m_glm <- cv.glmnet(X, factor(y), alpha = 0, family = "binomial", type.measure = "auc",
                   parallel = T, standardize = T, nfolds = 4, nlambda = 50)
  result[[target]] <- predict(m_glm, newx = X_test, type = "response", s = "lambda.min")
}
round(result, 1)
confusionMatrix(test[-1], result, positive = "1")
plot(cvfit)
result
