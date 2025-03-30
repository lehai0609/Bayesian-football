
setwd()

library(xml2)
library(RSQLite)
library(magrittr)
library(viridis)
library(tibble)
library(purrr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(pander)
library(tidyverse)

panderOptions('table.split.table', Inf)

# read the data
con = dbConnect(drv=RSQLite::SQLite(), dbname="Raw data and code/database.sqlite")

alltables = dbListTables(con)
tables <- alltables[alltables != "sqlite_sequence"]

lDataFrames <- vector("list", length=length(tables))


# create a data.frame for each table
for (i in seq(along=tables)) {
  lDataFrames[[i]] <- dbGetQuery(conn=con, statement=paste("SELECT * FROM '", tables[[i]], "'", sep=""))
}
matchdetails <- lDataFrames[[3]]


# choose which league and season to work on
db  <- src_sqlite("database.sqlite")
matches <- tbl(db, 'Match') %>% filter(country_id == 1729) # this is for EPL


# Loop over all matches and put the info into a dataframe

value_from_xpath  <- function(element, xpath, to.int = F, index = 1) {
  xml_find_all(element, xpath) %>%
    {ifelse(length(.), xml_text(.[[index]]), NA)} %>%
    {ifelse(to.int, as.integer(.), .)}
}

node_to_dataframe <- function(n, key) {
  tibble_(list(
    id = ~value_from_xpath(n, './id', to.int = T),
    type = ~value_from_xpath(n, './type'),
    subtype1 = ~value_from_xpath(n, './subtype'),
    subtype2 = ~value_from_xpath(n, paste0('./', key, '_type')),
    player1 = ~value_from_xpath(n, './player1'),
    player2 = ~value_from_xpath(n, './player2'),
    team = ~value_from_xpath(n, './team'),
    lon = ~value_from_xpath(n, './coordinates/value', to.int = T, index = 1),
    lat = ~value_from_xpath(n, './coordinates/value', to.int = T, index = 2),
    elapsed = ~value_from_xpath(n, './elapsed', T),
    elapsed_plus = ~value_from_xpath(n, './elapsed_plus', T)
  ))
}

# note: list of events: 'goal','shoton','shotoff','foulcommit','card','cross','corner','possession'
# you can choose whichever events you want to use in your analysis

incidents <- map_df(list('goal','shoton','shotoff','foulcommit','card','cross','corner','possession'), function(key) {
  matches %>%
    filter_(paste0('!is.na(', key, ')')) %>% 
    select_('id', key) %>%
    collect() %>% 
    rename_('value' = key) %>%
    pmap_df(function(id, value) {
      xml <- read_xml(value)
      df  <- xml %>%
        xml_find_all(paste0('/', key, '/value')) %>%
        map_df(node_to_dataframe, key)
      
      # Add the id of the game as 'foreign key' game_id
      if (nrow(df) > 0){
        if (length(xml_find_all(xml, paste0('/', key, '/value/', key, '_type'))) > 0) {
          df %<>%
            rename(tmp = subtype1) %>%
            rename(subtype1 = subtype2) %>%
            rename(subtype2 = tmp)
        }
        df$game_id  <- id
      }
      return(df)
    })
})

eplmatches = matchdetails %>% filter(league_id == 1729)

write.csv(eplmatches,"EPL_data/matchdetails.csv",row.names = F)
write.csv(incidents,"EPL_data/all_incidents.csv",row.names = F)
write.csv(lDataFrames[[4]],"EPL_data/players.csv",row.names = F)
write.csv(lDataFrames[[5]],"EPL_data/player_details.csv",row.names = F)
write.csv(lDataFrames[[6]],"EPL_data/teams.csv",row.names = F)