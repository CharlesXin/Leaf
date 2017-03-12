setwd("\\\\CSIADSRM01\\Public\\FTP Survey 2007\\Smart Control\\cxin\\Leaf Submit")
set.seed(1)
library(caret)

Submit = list()

for(i in seq(0,29)){
  Submit[[paste0("submit", i, seq="")]] = read.csv(paste0("submit", i, ".csv", seq=""))
}

pred_type_sum = data.frame(sapply(Submit, function(x){
  pred_index = apply(x[, -1], 1, which.max)
  pred_type = colnames(x)[pred_index+1]
  return(pred_type)
}))

pred_type_sum$id = Submit[[1]]$id

problem_samples = pred_type_sum[apply(pred_type_sum[, -ncol(pred_type_sum)], 1, function(x){
  length(unique(x))
})>1, ]

problem_samples_table = apply(problem_samples[, -ncol(problem_samples)], 1, table)

names(problem_samples_table) = as.character(problem_samples$id)


res = Submit[[1]]
for(i in 2:30){
  res = res + Submit[[i]]
}
res = res/30


# log(10^-15)/594 # 0.05814609
# log(0.5)/594 # 0.001166914
# (log(10^-15) - log(0.05))/594 # 0.05310277


res$max = apply(res[, -1], 1, max)
potential_prob_sample = res[res$max < 0.98, "id"] # 30 

potential_prob_sample_table = list()
for(i in potential_prob_sample){
  potential_prob_sample_table[[as.character(i)]] = colnames(res)[2:100][res[res$id == i, 2:100] > 0.005]
}


potential_prob_sample_predicted = list()
for(i in potential_prob_sample){
  potential_prob_sample_predicted[[as.character(i)]] = colnames(res)[2:100][res[res$id == i, 2:100] == res[res$id == i, "max"]]
}


potential_prob_sample_table = do.call(qpcR:::cbind.na, potential_prob_sample_table)
write.csv(potential_prob_sample_table, file = "potential_prob_sample_table.csv", row.names = F)

potential_prob_sample_predicted = do.call(c, potential_prob_sample_predicted)
potential_prob_sample_predicted = data.frame(t(potential_prob_sample_predicted))
colnames(potential_prob_sample_predicted) = potential_prob_sample
write.csv(potential_prob_sample_predicted, file = "potential_prob_sample_predicted.csv", row.names = F)

# wrong_id = c(297, 301, 887, 1304, 1382, 1387)
# wrong_res = res[res$id %in% wrong_id, ]
# wrong_res$max




