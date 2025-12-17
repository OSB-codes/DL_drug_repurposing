#!/usr/bin/env Rscript

jaccard <- function(x, y){
    sum(x %in% y) / length(unique(c(x,y)))
}
computeJaccardForDM <- function(preds, DM){
    Ja = rep(NA,ncol(DM))
    for(i in 1:length(Ja)){ Ja[i] = jaccard(preds, names(which(DM[,i]==1))) }
    return(Ja)
}
getOverlappingTargets <- function(preds, DM){
    Ta = cbind(rep(NA,ncol(DM)), rep(NA,ncol(DM)))
    for(i in 1:nrow(Ta)){ 
      ove = intersect(preds, names(which(DM[,i]==1)))
      Ta[i,1] = paste0(ove,collapse=';')
      Ta[i,2] = paste0(names(which(DM[,i]==1))[!names(which(DM[,i]==1)) %in% ove],collapse=";")  
    }
    colnames(Ta) = c('Targeted','Untargeted')
    return(Ta)
}
hypergeo_pval <- function(overlap, total_proteins, drug_targets, predicted_proteins) {
  m <- drug_targets
  n <- total_proteins - m
  k <- predicted_proteins
  q <- overlap - 1
  pval <- phyper(q, m, n, k, lower.tail = FALSE)
  return(pval)
}
computeHypergeoForDM <- function(preds, DM, num_total_proteins){ # 20455
    Hy = rep(NA,ncol(DM))
    num_predicted_proteins = length(preds)
    for(i in 1:length(Hy)){ 
        drug_targets <- length(names(which(DM[,i]==1)))    
        overlap <- sum(preds %in% names(which(DM[,i]==1)))
        Hy[i] = hypergeo_pval(overlap, num_total_proteins, drug_targets, num_predicted_proteins)
    }
    return(Hy)
}

args = commandArgs(TRUE) 
infile = args[1] 
outdir = './'
nam = gsub("^(.+)\\.txt","\\1",infile)
outfile = paste0(nam,"_drug_pvalues.csv")

library(Matrix)
drug_category = readRDS(file='reference_drugs.rds')
targets = read.table("targetList.txt",sep="\t",header=FALSE,stringsAsFactors=FALSE); targets = targets[,1]
subDM = readRDS('subDM_pert_target_all_human_Matrix.rds')

# Ranndomization
ranfiles = list.files(pattern=paste0(gsub("_local","",nam),"__\\d+.*\\.txt$"),path='randoms',full.name=TRUE)

lines <- readLines(infile, warn = FALSE)
non_empty_lines <- lines[grepl("\\S", lines)]
if (length(non_empty_lines) == 0) {
  message("Input file is effectively empty (no meaningful lines). Skipping...")
  preds <- NULL  # or handle however you'd like
} else {
  preds <- read.table(infile, sep = "\t", header = FALSE, stringsAsFactors = FALSE, quote = ""); preds = preds[,1]
}
preds = preds[preds %in% targets]; #tar1 %in% preds

# 1. hypergeomtric p-value
Hy = computeHypergeoForDM(preds, subDM, 20455)

# 2. empirical p-value 
JaRs = matrix(0,nrow=ncol(subDM), ncol=length(ranfiles))
for(i in 1:length(ranfiles)){
  f = ranfiles[i]; print(f)
  temp = read.csv(f,header=FALSE)
  temp = temp[,1]
  jar = computeJaccardForDM(temp, subDM)
  jar[is.nan(jar)] = 0
  JaRs[,i] = jar
}
rownames(JaRs) = colnames(subDM)
pemp_JaRs = rowSums(JaRs >0) / ncol(JaRs) # random 'hit'

# zscore
pval_to_z <- function(pval) {
  if (pval == 0) {
    return(Inf)
  } else {
    return(-qnorm(pval / 2))
  }
}
hy_z_scores = sapply(Hy, pval_to_z)
sum(hy_z_scores >=2)

S = cbind('pemp_JaRs'=pemp_JaRs)
rownames(S) = colnames(subDM)
S = cbind(S,'hyergeoPvalue'=Hy, 'hy_z_scores'=hy_z_scores)
S = cbind(S,'rank_hy_z_scores'=rank(-as.numeric(S[,'hy_z_scores']),ties.method='min'))
S = cbind(S,'Functional.Annotation'=drug_category[match(rownames(S), drug_category[,1]),c('Functional.Annotation','Category')])
S = cbind(S, getOverlappingTargets(preds, subDM))
S <- data.frame(Compound=rownames(S), S)
write.table(S,file=outfile,sep="\t",row.names=FALSE,col.names=TRUE,quote=TRUE)

