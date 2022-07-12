import pandas as pd


dfmi = pd.DataFrame([list('abcd'),
                     list('efgh'),
                     list('ijkl'),
                     list('mnop')],
                    columns=pd.MultiIndex.from_product([['one', 'two'],
                                                        ['first', 'second']]))

dfmi['one']['second'].iloc[0] = 'a'
dfmi.loc[:, ('one', 'second')].iloc[0] = 'a'