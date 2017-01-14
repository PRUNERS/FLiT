CREATE OR REPLACE FUNCTION public.createschmoo(run integer,
                                               prec text[],
                                               compilers text[],
                                               fname text
)
 RETURNS text
 LANGUAGE plpython3u
AS $function$
from plpy import spiexceptions
from sys import path
path.append('/home/sawaya/temp/qfp/db/python')
import plotting as pl

prec_str = ""
if len(prec) > 0:
   prec_str = " and (precision = '"
   for t in prec:
      prec_str += t + "' or precision = '"
   prec_str = prec_str[:-17] + ")"

comp_str = ""
if len(compilers) > 0:
   comp_str = " and (compiler = '"
   for c in compilers:
      comp_str += c + "' or compiler = '"
   comp_str = comp_str[:-16] + ")"
   
quer = ("select distinct name from tests as t1 where exists " +
        "(select 1 from tests where t1.name = name and t1.precision " +
        "= precision and t1.score0 != score0 and t1.run = run " +
        "and t1.compiler = compiler) " +
        "and run = " + str(run) + prec_str + comp_str +
        " order by name")
tests = plpy.execute(quer)

tests_str = ""
if len(tests) > 0:
   tests_str = " and (name = '"
   for t in tests:
      tests_str += t['name'] + "' or name = '"
   tests_str = tests_str[:-12] + ")"

quer = ("select distinct switches, compiler, precision " +
        "from tests where " +
        "run = " + str(run) +
        prec_str + comp_str + tests_str +
        " order by compiler, switches")
x_axis = plpy.execute(quer)
xa_count = len(x_axis)

x_ticks = []
y_ticks = []
z_data = []

x_count = 0
y_count = 0

ret_val = ""

for x in x_axis:
   x_ticks.append(x['switches'])
scores = []
for t in tests:
   y_ticks.append(t['name'])
   y_count += 1
   del scores[:]
   quers = ("select distinct score0, switches, " +
            "compiler, optl from " +
            " tests where run = " + str(run) +
            " and name = '" + t['name'] + "'" + prec_str + comp_str +
            " and optl = '-O0'" +
            " order by compiler, optl, switches")
   scores.append(plpy.execute(quers))
   quers = ("select distinct score0, switches, " +
            "compiler, optl from " +
            " tests where run = " + str(run) +
            " and name = '" + t['name'] + "'" + prec_str + comp_str +
            " and optl = '-O1'" +
            " order by compiler, optl, switches")
   scores.append(plpy.execute(quers))
   quers = ("select distinct score0, switches, " +
            "compiler, optl from " +
            " tests where run = " + str(run) +
            " and name = '" + t['name'] + "'" + prec_str + comp_str +
            " and optl = '-O2'" +
            " order by compiler, optl, switches")
   scores.append(plpy.execute(quers))
   quers = ("select distinct score0, switches, " +
            "compiler, optl from " +
            " tests where run = " + str(run) +
            " and name = '" + t['name'] + "'" + prec_str + comp_str +
            " and optl = '-O3'" +
            " order by compiler, optl, switches")
   scores.append(plpy.execute(quers))
   eq_classes = []
   del eq_classes[:]
   line_classes = []
   for set in scores:
      color = 0
      eq = {}
      for x in set:
         if not x['score0'] in eq:
            eq[x['score0']] = color
            color += 1
      eq_classes.append(eq)
   for x in x_axis:
      quer = ("select score0, optl from tests where name = '" +
              t['name'] + "' and precision = '" + x['precision'] +
              "' and switches = '" + x['switches'] +
              "' and compiler = '" + x['compiler'] +
              "' and run = " + str(run) + " order by optl")
      score = plpy.execute(quer)
      assert len(score) == 4, quer
      x_count += 1
      test_scores = []
      try:
         for de in zip(score, eq_classes):
            test_scores.append(de[1][de[0]['score0']])
      except KeyError:
         return ("key error fetching color: " + quer + " " + quers + "**" +
                 str(eq_classes))
      line_classes.append(test_scores)
   z_data.append(line_classes)

pl.plot(x_ticks, y_ticks, z_data, fname, ', '.join(compilers) +
        ' @ precision(s): ' +
        ', '.join(prec))

return str(len(z_data))

$function$
