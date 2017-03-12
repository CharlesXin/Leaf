setwd("\\\\CSIADSRM01\\Public\\FTP Survey 2007\\Smart Control\\cxin\\submit")

Submit = list()
for(i in seq(0,29)){
  Submit[[paste0("submit", i, seq="")]] = read.csv(paste0("submit", i, ".csv", seq=""))
}

res = Submit[[1]]
for(i in 2:30){
  res = res + Submit[[i]]
}
res = res/30

Manual_use = read.csv("Maual_Use.csv")
for( i in 1:nrow(Manual_use)){
  
  id = Manual_use[i, "id"]
  type = as.character(Manual_use[i, "Type"])
  
  print(res[res$id == id, type])
  
  res[, -1][res$id == id, ] = rep(0, 99) 
  res[res$id == id, type] = 1
  
}

res[, -1] = round(res[, -1], 1)

write.csv(res, file = "Submit_Final.csv", row.names = F)

#####################################################################################################################

