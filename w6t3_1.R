library(keras) 

imbd <- dataset_imdb(num_words = 10000) # import data
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imbd # make test and training data and label

vectorize_sequences <- function(sequences, dimension = 10000){ # make function to vectorize sequences
  results <- matrix(0, nrow = length(sequences), ncol = dimension) # create matrix with a row for each observation and column for each data point
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1 # cyle through rows for each observation i and add a 1 in the column representing the value of that observation
  results
}

x_train <- vectorize_sequences(train_data) # vect. training set
x_test <- vectorize_sequences(test_data) # vect. test set

y_train <- as.numeric(train_labels) # convert labels to numeric
y_test <- as.numeric(test_labels) # convert labels to numeric

model <- keras_model_sequential() %>% # build model
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile( # choose loss function and optimizer
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

## create validation set

val_indices <- 1:10000
x_val <- x_train[val_indices, ]
partial_x_train  <- x_train[-val_indices, ]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

plot(history)

## make predictions

model %>% predict(x_test[1:10,])
