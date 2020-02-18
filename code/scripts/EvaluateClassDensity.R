args <- commandArgs(trailingOnly = TRUE)
inputCsv <- args[1]
df <- read.csv(inputCsv)
N <- nrow(df)
print(paste0('Read metadata rows for ',N,' trainig images'))

df$grapheme_root <- as.factor(df$grapheme_root)
df$vowel_diacritic <- as.factor(df$vowel_diacritic)
df$consonant_diacritic <- as.factor(df$consonant_diacritic)

require('plyr')

fields <- c('grapheme_root','vowel_diacritic','consonant_diacritic')
outArgIdx <- 2
for(field in fields) {
  statsDf <- count(df,field)
  statsDf$prob <- statsDf$freq / N
  statsDf <- statsDf[order(statsDf$freq),]
  outFile <- args[outArgIdx]
  write.csv(statsDf, file=outFile, row.names = F, quote = F)
  print(paste0('Wrote ',outFile))
  outArgIdx <- outArgIdx+1
}
print('Done')