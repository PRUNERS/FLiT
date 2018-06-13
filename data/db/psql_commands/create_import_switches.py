CREATE OR REPLACE FUNCTION importSwitches(path text) RETURNS integer as $$
with open(path) as inf:
    count = 0
    for line in inf:
        spc = line.find(' ')
        if spc == -1:
            abbrev = line
            swts = ''
        else:
            abbrev = line[0:spc]
            swts = line[spc+1:-1]
        q = ("INSERT INTO switch_conv VALUES " +
             "('" + abbrev + "', '" + swts + "')")
        plpy.execute(q)
        count = count + 1
return count
$$ LANGUAGE plpython3u;
