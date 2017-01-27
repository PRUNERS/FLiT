## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings)
## -- NOTE: first variable listed will be x1, remainder will be parents, x2, ..., xk
## probs: vector of probabilities for the flattened probability table
## levelsList: a list containing a vector of levels (outcomes) for each variable
## See the BayesNetExamples.r file for examples of how this function works
createCPT = function(varnames, probs, levelsList)
{
  ## Check dimensions agree
  if(length(probs) != prod(sapply(levelsList, FUN=length)))
    return(NULL)

  ## Set up table with appropriate dimensions
  m = length(probs)
  n = length(varnames)
  g = matrix(0, m, n)

  ## Convert table to data frame (with column labels)
  g = as.data.frame(g)
  names(g) = varnames

  ## This for loop fills in the entries of the variable values
  k = 1
  for(i in n:1)
  {
    levs = levelsList[[i]]
    g[,i] = rep(levs, each = k, times = m / (k * length(levs)))
    k = k * length(levs)
  }

  return(data.frame(probs = probs, g))
}

## Build a CPT from a data frame
## Constructs a conditional probability table as above, but uses frequencies
## from a data frame of data to generate the probabilities.
createCPT.fromData = function(x, varnames)
{
  levelsList = list()

  for(i in 1:length(varnames))
  {
    name = varnames[i]
    levelsList[[i]] = sort(unique(x[,name]))
  }

  m = prod(sapply(levelsList, FUN=length))
  n = length(varnames)
  g = matrix(0, m, n)

  ## Convert table to data frame (with column labels)
  g = as.data.frame(g)
  names(g) = varnames

  ## This for loop fills in the entries of the variable values
  k = 1
  for(i in n:1)
  {
    levs = levelsList[[i]]
    g[,i] = rep(levs, each = k, times = m / (k * length(levs)))
    k = k * length(levs)
  }

  ## This is the conditional probability column
  probs = numeric(m)
  numLevels = length(levelsList[[1]])
  skip = m / numLevels

  ## This chunk of code creates the vector "fact" to index into probs using
  ## matrix multiplication with the data frame x
  fact = numeric(ncol(x))
  lastfact = 1
  for(i in length(varnames):1)
  {
    j = which(names(x) == varnames[i])
    fact[j] = lastfact
    lastfact = lastfact * length(levelsList[[i]])
  }
  ## Compute unnormalized counts of subjects that satisfy all conditions
  a = as.matrix(x - 1) %*% fact + 1
  for(i in 1:m)
    probs[i] = sum(a == i)

  ## Now normalize the conditional probabilities
  for(i in 1:skip)
  {
    denom = 0 ## This is the normalization
    for(j in seq(i, m, skip))
      denom = denom + probs[j]
    for(j in seq(i, m, skip))
    {
      if(denom != 0)
        probs[j] = probs[j] / denom
    }
  }

  return(data.frame(probs = probs, g))
}

# This is almost the same as the function above but implemented differently
# Note: the first name in varnames is considered the dependent variable
#   which is conditioned on all of the other varnames.
# This implementation is slower than the one above, but it is easier
# for me to understand.
createCPT.fromData2 = function(x, varnames)
{
  n = nrow(x)
  joint = aggregate(list(probs = rep(1/n, n)), by=x[varnames], FUN='sum')
  others = marginalizeFactor(joint, varnames[1])
  names(others)[names(others) == 'probs'] = 'probs.others'
  merged = merge(joint, others, by=varnames[-1])
  merged$probs = merged$probs / merged$probs.others
  merged = merged[c('probs', varnames)]
  return(merged)
}

## Rows of A that match the columns and values of frameRow
## A: A frame
## toMatch: A list of name -> value pairs
## namesToMatch: a list of corresponding names if toMatch is a vector
##
## All of the columns of frameRow need to exist in A since they are all
## used to check which row numbers are to be returned.
matchingRows = function(A, toMatch)
{
  rows = TRUE
  for (name in names(toMatch))
    rows = rows & (A[name] == toMatch[[name]])
  which(rows)
}

## Product of two factors
## A, B: two factor tables
##
## Should return a factor table that is the product of A and B.
## You can assume that the product of A and B is a valid operation.
productFactor = function(A, B)
{
  # Rename the column name for probs in both tables
  names(A)[names(A) == 'probs'] = 'probs.A'
  names(B)[names(B) == 'probs'] = 'probs.B'

  # Find the common column names
  commonNames = setdiff(intersect(names(A), names(B)), c('probs'))

  # Merge the two tables and product their probabilities
  merged = merge(A, B, by=commonNames)
  merged$probs = merged$probs.A * merged$probs.B

  # Reorder columns and throw away the probs.A and probs.B columns
  columns = setdiff(names(merged), c('probs', 'probs.A', 'probs.B'))
  columns = c('probs', columns)
  merged = merged[columns]
  return(merged)
}

## Marginalize a variable from a factor
## A: a factor table
## margVar: a string of the variable name to marginalize
##
## Should return a factor table that marginalizes margVar out of A.
## You can assume that margVar is on the left side of the conditional.
marginalizeFactor = function(A, margVar)
{
  remainingVars = setdiff(names(A), c('probs', margVar))
  if (length(remainingVars) > 0 )
  {
    marginalized = aggregate(A['probs'], by = A[remainingVars], FUN = 'sum')
    # Reorder columns
    marginalized = marginalized[c('probs', setdiff(names(marginalized), 'probs'))]
  }
  else
    marginalized = data.frame(probs = 1)
  return(marginalized)
}

## Marginalize a list of variables
## bayesnet: a list of factor tables
## margVars: a vector of variable names (as strings) to be marginalized
##
## Should return a Bayesian network (list of factor tables) that results
## when the list of variables in margVars is marginalized out of bayesnet.
marginalize = function(bayesnet, margVars)
{
  for (i in seq_len(length(margVars)))
  {
    # Create a map of (margvar) -> (indexes of tables having margvar)
    varTblIdxMap = sapply(
      margVars,                            # One entry per margVar
      function (x) as.vector(which(        # Convert list of bools to vector of ints where TRUE
        sapply(                            # Tally which tables have margVar
          bayesnet,
          function (y) x %in% names(y)     # margVar %in% names(tbl) is a single TRUE or FALSE
          )
        )),
      simplify = FALSE                     # Keep the result as a list, not a vector
      )
    # Sort the map by the number of tables containing that variable
    varTblIdxMap = varTblIdxMap[order(sapply(varTblIdxMap, length))]

    # Take the top one
    margVar = names(varTblIdxMap)[1]
    tableIdx = varTblIdxMap[[1]]

    # Remove the margVar from the list
    idx = which(margVars == margVar)
    margVars = margVars[-idx]

    # If there are tables to marginalize for this variable, then do it
    if (length(tableIdx) > 0)
    {
      # Grab the tables out and remove them from the bayesnet
      # Remove the tables from the bayesnet
      tables = bayesnet[tableIdx]
      bayesnet = bayesnet[-tableIdx]

      # Merge the tables together to one joint table
      merged = Reduce(productFactor, tables)

      # Marginalize the factor out
      merged = marginalizeFactor(merged, margVar)

      # Add the marginalized joint dist back to the bayesnet
      bayesnet[[length(bayesnet) + 1]] = merged
    }
  }
  return(bayesnet)
}

## Observe values for a set of variables
## bayesnet: a list of factor tables
## obsVars: a vector of variable names (as strings) to be observed
## obsVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the observed variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
## to be probability mass functions.
observe = function(bayesnet, obsVars, obsVals)
{
  # Construct a list of var -> val
  toFind = as.list(obsVals)
  names(toFind) = obsVars

  # For each table, only keep the rows that match toFind
  for(i in 1:length(bayesnet))
  {
    tbl = bayesnet[[i]]

    subVars = obsVars[obsVars %in% names(tbl)]
    subToFindNames = intersect(names(toFind), names(tbl))
    if (length(subToFindNames) > 0)
    {
      rowsToKeep = matchingRows(tbl, toFind[subToFindNames])
      bayesnet[[i]] = tbl[rowsToKeep,]
    }
  }
  return(bayesnet)
}

## Run inference on a Bayesian network
## bayesnet: a list of factor tables
## margVars: a vector of variable names to marginalize
## obsVars: a vector of variable names to observe
## obsVals: a vector of values for corresponding variables (in the same order)
##
## This function should run marginalization and observation of the sets of
## variables. In the end, it should return a single joint probability table. The
## variables that are marginalized should not appear in the table. The variables
## that are observed should appear in the table, but only with the single
## observed value. The variables that are not marginalized or observed should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
infer = function(bayesnet, margVars, obsVars, obsVals)
{
  # Observe & Marginalize
  bayesnet = observe(bayesnet, obsVars, obsVals)
  bayesnet = marginalize(bayesnet, margVars)

  # Combine the remaining tables
  joint = Reduce(productFactor, bayesnet)

  # Normalize the table
  joint$probs = joint$probs / sum(joint$probs)

  return(joint)
}
