import cPickle
import scipy
import shelve

--- pickle ---

atuple = (4,5,6)

cPickle.dumps(atuple)

cPickle.loads(<string>)

cPickle.loads(cPickledumps(atuple))

cPickle.dump(atuple, open('mydata.pkl','w'))

cPickle.load(open('mydata.pkl'))

--- shelve --- 

sh = shelve.open('myshelve.dat')

sh['fred'] = 'joe'
sh['t'] = (2,3)
sh.close()

ds = shelve.open('myshelve.dat')

---

cPickle.load