# -- LICENSE BEGIN --
#
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# Written by
#   Michael Bentley (mikebentley15@gmail.com),
#   Geof Sawaya (fredricflinstone@gmail.com),
#   and Ian Briggs (ian.briggs@utah.edu)
# under the direction of
#   Ganesh Gopalakrishnan
#   and Dong H. Ahn.
#
# LLNL-CODE-743137
#
# All rights reserved.
#
# This file is part of FLiT. For details, see
#   https://pruners.github.io/flit
# Please also read
#   https://github.com/PRUNERS/FLiT/blob/master/LICENSE
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
#
# - Redistributions in binary form must reproduce the above
#   copyright notice, this list of conditions and the disclaimer
#   (as noted below) in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the LLNS/LLNL nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL
# SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# Additional BSD Notice
#
# 1. This notice is required to be provided under our contract
#    with the U.S. Department of Energy (DOE). This work was
#    produced at Lawrence Livermore National Laboratory under
#    Contract No. DE-AC52-07NA27344 with the DOE.
#
# 2. Neither the United States Government nor Lawrence Livermore
#    National Security, LLC nor any of their employees, makes any
#    warranty, express or implied, or assumes any liability or
#    responsibility for the accuracy, completeness, or usefulness of
#    any information, apparatus, product, or process disclosed, or
#    represents that its use would not infringe privately-owned
#    rights.
#
# 3. Also, reference herein to any specific commercial products,
#    process, or services by trade name, trademark, manufacturer or
#    otherwise does not necessarily constitute or imply its
#    endorsement, recommendation, or favoring by the United States
#    Government or Lawrence Livermore National Security, LLC. The
#    views and opinions of authors expressed herein do not
#    necessarily state or reflect those of the United States
#    Government or Lawrence Livermore National Security, LLC, and
#    shall not be used for advertising or product endorsement
#    purposes.
#
# -- LICENSE END --

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
