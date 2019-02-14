library(keras)
library(tidyverse)
library(DataExplorer)
library(rsample)
library(recipes)

data_raw = read.csv(file="/home/neo/Downloads/Higgs.csv", header=TRUE, sep=",")
glimpse(data_raw)

data_raw = data_raw %>% select(Label, everything())

plot_missing(data_raw)

seed(2019)
train_test_split = initial_split(data_raw, prop=0.8)
train_test_split

train_data = training(train_test_split) %>% select(-EventId)
test_data = testing(train_test_split) %>% select(-EventId)

glimpse(train_data)

rec_obj = recipe(Label ~ ., data=train_data) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data=train_data)

train_x = bake(rec_obj, new_data=train_data) %>% select(-Label)
test_x = bake(rec_obj, new_data=test_data) %>% select(-Label)

train_y = ifelse(train_data$Label=="s",1,0)
train_y
test_y = ifelse(test_data$Label=="s",1,0)
test_y

model = keras_model_sequential() %>% 
  layer_dense(units=128, activation="relu", initializer_he_normal(), input_shape=ncol(train_x)) %>% 
  layer_dense(units=56, activation="relu", initializer_he_normal()) %>% 
  layer_dense(units=16, activation="relu", initializer_he_normal()) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units=1, activation="sigmoid")

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(as.matrix(train_x), as.numeric(train_y), epochs=5, batch_size=128, validation_split=.2)

metric = model %>% evaluate(as.matrix(test_x), as.numeric(test_y))
metric

  
  
  
  
  
  
  
  
  

