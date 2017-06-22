install.packages("RPostgreSQL")
require("RPostgreSQL")

drv <- dbDriver("PostgreSQL")

con <- dbConnect(drv, dbname = "flit")

                                        #sanity test
dbExistsTable(con, "tests")

                   #first, we'll get the column (variable) names
db_columnNames <- dbGetQuery(con, "select name from opcodes order by index")
db_pcolumnNames <- dbGetQuery(con, "select concat('pred_', name) as name from opcodes order by index")

db_opcounts <- dbGetQuery(con, "select concat(switches, '_', precision, '_', name) as switches, array_to_string(array(select coalesce(count,0) from opcodes left join op_counts on (opcodes.index = op_counts.opcode and op_counts.test_id = tests.index and dynamic=true) order by opcodes.index), ' ') count, array_to_string(array(select coalesce(pred_count,0) from opcodes left join op_counts on (opcodes.index = op_counts.opcode and op_counts.test_id = tests.index and dynamic=true) order by opcodes.index), ' ') pcount from tests where run = 16 and host = 'kingspeak' and compiler = 'icpc' and exists (select 1 from op_counts where test_id = tests.index)")

## db_opcounts <- dbGetQuery(con, "select concat(switches, '_', precision, '_', name) as switches, array_to_string(array(select coalesce(count,0) from opcodes left join op_counts on (opcodes.index = op_counts.opcode and op_counts.test_id = tests.index and dynamic=true) order by opcodes.index), ' ') as count, array_to_string(array(select coalesce(pred_count,0) from opcodes left join op_counts on (opcodes.index = op_counts.opcode and op_counts.test_id = tests.index and dynamic=true) order by opcodes.index), ' ') as pcount from tests where run = 16 and host = 'kingspeak' and compiler = 'icpc'")

data <- matrix(nrow = length(db_opcounts[,1]), ncol = 2 * length(db_columnNames[,1]), byrow = TRUE,
               dimnames = list(unlist(db_opcounts["switches"]),
               mapply(c, db_columnNames["name"], db_pcolumnNames["name"])))

for (i in 1:length(unlist(db_opcounts["count"]))){
    data[i,] = mapply(c, strsplit(db_opcounts["count"][i,], " "),
            strsplit(db_opcounts["pcount"][i,], " "))
}

#convert our strings to numbers (seemingly necessary with the arrays from query)
class(data) <- "numeric"

#remove the zero columns -- unable to scale data sets with these
data2 <- data[ , !apply(data==0,2,all)]

#also must remove constant columns
data3 <- data2[,apply(data2, 2, var, na.rm=TRUE) != 0]

#generate PCs (PCA)
pc <- prcomp(data3, scale.=TRUE)

plot(pc)

plot(pc, type='l')

summary(pc)

#gather PCs (we'll use first 4 -- #1 dominates seriously, might try a log scaling later)
comp <- data.frame(pc$x[,1:4])

plot(comp, pch=16, col=rgb(0,0,0,0.5))

#base variance
wss <- (nrow(comp)-1)*sum(apply(comp,2,var))

for (i in 2:15) wss[i] <- sum(kmeans(comp,
                                     centers=i, nstart=25, iter.max=1000)$withinss)

plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

#we'll try 6 clusters, based on previous output
k <- kmeans(comp, 6, nstart=25, iter.max=1000)

library(RColorBrewer)
library(scales)
palette(alpha(brewer.pal(9, 'Set1'), 0.5))
plot(comp, col=k$clust, pch=16)


                                        #dump the list by cluster

sort(table(k$clust))
clust <- names(sort(table(k$clust)))

row.names(data[k$clust==clust[1],])

row.names(data[k$clust==clust[2],])

row.names(data[k$clust==clust[3],])

row.names(data[k$clust==clust[4],])

row.names(data[k$clust==clust[5],])

row.names(data[k$clust==clust[6],])



dbDisconnect(con)
