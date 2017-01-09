CREATE OR REPLACE FUNCTION public.createSchmoo(run integer, precision text)
RETURNS text
LANGUAGE plpython3u
AS $function$
import plotly.plotly as py
import plotly.graph_objs as go

quer = ("select distinct switches, compiler, optl from tests where " +
        "run = " + str(run) +
        "and precision = '" + precision + "'")
x_axis = plpy.execute(quer)

quer = ("select name from tests where run = " + str(run) +
        " and precision = '" + precision + "'")

y_axis = plpy.execute(quer)

xes = []
yes = []

for each t in y_axis:
    quer = ("select distinct score0d from tests where run = " + str(run) +
            " and name = '" + t['name'] + "' and precision = '" +
            precision + "'")
    scores = plpy.execute(quer)
    eq_classes = {}
    for x in range(0, len(scores)):
        eq_classes[scores[x]['score0d']] = x
    for x in range(0, len(x_axis)):
        xes.append(x)
        quer = ("select score0d from tests where name = '" +
                t['name'] + "' and precision = '" + precision +
                "' and switches = '" + x['switches'] +
                "' and compiler = '" + x['compiler'] +
                "' and optl = '" + x['optl'] + "'")
        score = plpy.execute(quer)
        yes.append(eq_classes[x['score0d']])

trace = go.Scatter(
    x = xes,
    y = yes,
    mode = 'markers'
)

data = [trace]

return py.plot(data, filename='basic-line')

$function$
