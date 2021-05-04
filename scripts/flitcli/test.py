import flit_analyze as az

root = az.Event.create_root_event(0,100)

event1 = az.Event()
event1.name = 'ev1'
event1.duration = 25
event1.properties['key'] = str([n for n in range(25)])

event2 = az.Event()
event2.name = 'ev2'
event2.duration = 25
root.children.append(event2)

event3 = az.Event()
event3.name = 'ev3'
event3.duration = 50
root.children.append(event3)
event3.children.append(event1)

event4 = az.Event()
event4.name = 'ev4'
event4.duration = 15
event3.children.append(event4)

event5 = az.Event()
event5.name = 'ev5'
event5.duration = 5
root.children.append(event5)

event6 = az.Event()
event6.name = 'ev6'
event6.duration = 35
root.children.append(event6)

str = az.tree_toString(root)[1:]
print(str)
