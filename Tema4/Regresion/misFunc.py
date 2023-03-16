#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Función que calcula el IV
def calcularIV(var_categoricas,var_target,datos):
    import numpy as np
    resultado=[] #Array resultado
    
    for v_cat in var_categoricas:
        var_target = np.array(var_target)
        var_values = np.array(datos[v_cat])
        var_levels = np.unique(var_values)

        mat_values = np.zeros(shape=(len(var_levels),2))

        for i in range(len(var_target)):
            # Obtención de la posición en los niveles del valor
            for j in range(len(var_levels)):
                if var_levels[j] == var_values[i]:
                    pos = j
                    break

            # Estimación del número valores en cada nivel
            if var_target[i]:
                mat_values[pos][0] += 1
            else:
                mat_values[pos][1] += 1

            # Obtención del IV
            IV = 0
            for j in range(len(var_levels)):
                if mat_values[j][0] > 0 and mat_values[j][1] > 0:
                    rt = mat_values[j][0] / (mat_values[j][0] + mat_values[j][1])
                    rf = mat_values[j][1] / (mat_values[j][0] + mat_values[j][1])
                    IV += (rt - rf) * np.log(rt / rf)        
        # Se agrega el IV al listado
        resultado.append(IV)
    return resultado


# In[5]:


# Función que calcula el VIF
def calcularVIF(data):
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    
    features = list(data.columns)
    num_features = len(features)
    
    model = LinearRegression()
    
    result = pd.DataFrame(index = ['VIF'], columns = features)
    result = result.fillna(0)
    
    for ite in range(num_features):
        x_features = features[:]
        y_featue = features[ite]
        x_features.remove(y_featue)
        
        x = data[x_features]
        y = data[y_featue]
        
        model.fit(data[x_features], data[y_featue])
        
        result[y_featue] = 1/(1 - model.score(data[x_features], data[y_featue]))
    
    return result


# In[6]:


# Selecciona las variables con VIF menor al máximo
# Ejecuta el cáluculo del VIF de forma iterativa, eliminando una variable en cada iteración
# hasta que todas las variables tengan un VIF por debajo del punto de corte
def seleccionarPorVIF(data, max_VIF = 5):
    import numpy as np
    
    result = data.copy(deep = True)
    
    VIF = calcularVIF(result)
    
    while VIF.values.max() > max_VIF:
        col_max = np.where(VIF == VIF.values.max())[1][0]
        features = list(result.columns)
        features.remove(features[col_max])
        result = result[features]
        
        VIF = calcularVIF(result)
        
    return result


# In[ ]:


# features: Lista de nombres de columnas menos la target
# x: Columnas menos la target
# y: Target
def StepWise(features, x, y):
    from sklearn.linear_model import LinearRegression
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np

    # Modelo para realizar los ajustes
    model = LinearRegression()

    # Variable para almecena los índices de la lista de atributos usados
    feature_order =  []
    feature_error = []

    # Iteración sobre todas las variables
    for i in range(len(features)):
        idx_try = [val for val in range(len(features)) if val not in feature_order]
        iter_error = []

        for i_try in idx_try:
            useRow = feature_order[:]
            useRow.append(i_try)

            use = x[x.columns[useRow]]

            model.fit(use, y)
            rmsError = np.linalg.norm((y - model.predict(use)), 2)/np.sqrt(len(y))
            iter_error.append(rmsError)

        pos_best = np.argmin(iter_error)
        feature_order.append(idx_try[pos_best])
        feature_error.append(iter_error[pos_best])

    for i in range(len(features)):
        print("En el paso", i, "se ha insertado la variable", 
              features[feature_order[i]], "con un error", feature_error[i])
    
    plt.plot(range(len(features)), feature_error, 'r-', label = 'Datos')
    plt.xlabel('Numero de atributos')
    plt.ylabel('Error (RMS)')

