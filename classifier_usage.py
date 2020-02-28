import numpy as np
#import matplotlib.pyplot as plt
from sklearn import svm
import pickle
from string import punctuation
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time

# classiifer file name here
filename = "classifier_combined_data_json80.sav"

loaded_model = pickle.load(open(filename, 'rb'))

print("loading vectors...")
with open('/Users/tobycourtis/Desktop/Data/final_form_data/index_map.pkl', 'rb') as f:
    index_map = pickle.load(f)

vectors2 = np.load("/Users/tobycourtis/Desktop/Data/final_form_data/vectors-float32.npz")
vectors = vectors2['arr_0'][:]
print("Done loading vectors... ")


# test vectors if needed (length 300)
the = np.array([-0.20838, -0.14932001, -0.017527999, -0.028432, -0.060104001, -0.26460001, -4.1444998, 0.62932003, 0.33671999, -0.43395001, 0.39899001, -0.19573, 0.13977, -0.021519, 0.37823001, -0.55250001, -0.1123, -0.0081443004, 0.29058999, 0.066817001, 0.10465, -0.086943001, -0.048983, -0.26756999, -0.47038001, 0.27469, 0.069245003, -0.027967, -0.19719, 0.016749, -0.29681, 0.17838, 0.058373999, -0.24806, 0.085845999, 0.35043001, 0.049157001, -0.16430999, 0.50011998, -0.18053, 0.31422001, 0.10671, 0.031851999, 0.074277997, 0.27956, 0.080316998, 0.054779999, -0.30349001, -0.43215001, 0.32416999, 0.40856001, 0.36192, 0.13445, -0.12932999, 0.11331, -0.15755001, 0.35755, 0.30463001, -0.098488003, 0.012032, 0.45581001, 0.37101001, 0.1427, -0.43329, -0.10869, 0.49849001, 0.54455, 0.44352001, 0.31804001, 0.022171, -0.41185999, -0.025428001, 0.21062, -0.3583, 0.22028001, -0.55391002, -0.035363998, -0.053998001, 0.32172, -0.51928002, -0.27427, -0.45214, -0.329, -0.48519, 0.52965999, 0.0041434001, -0.1718, -0.18748, -0.24365, -0.060786001, 0.050733, -0.21335, 0.27627, 0.42745, 0.011461, -0.29793999, -3.2881, -0.39842001, 0.16796, -0.12894, 0.0020005, 0.45613, 0.15215001, 0.15364, -0.21280999, -0.25339001, 0.28955001, -0.57817, -0.6074, 0.10301, 0.28323999, 0.21506, -0.042325001, 0.40479001, -0.20579, -0.011674, 0.26091999, -0.15402, 0.057960998, -0.058575999, -0.41973999, 0.52015001, 0.15074, -0.088039003, -0.14445999, -0.17073999, 0.083751999, -0.25707999, 0.16362, 0.14794999, -0.059820998, 0.034472998, -0.14534, -0.17964999, 0.076302998, 0.33353999, -0.14433999, 0.17618001, 0.45344999, 0.15262, -0.075099997, 0.27592, 0.081455998, 0.30737999, -0.072327003, 0.10706, -0.35580999, -0.026690001, 0.61236, 0.70828998, -0.28944999, -0.024637001, 0.01189, -0.091899, -0.27272001, -0.10157, 0.44712999, 0.092418, -0.10711, -0.015552, 0.12822001, 0.22256, -0.069058999, 0.29927, -0.10913, 0.1618, 0.14796001, 0.1136, 0.26633999, 0.010832, 0.071946003, 0.16972999, -0.22769, 0.322, -0.083747998, 0.65268999, 0.068244003, -0.32686999, 0.31782001, 0.17035, 0.79803002, -0.19193999, -0.16485, -0.32437, 0.079104997, -0.35672, -0.26786, -0.24786, 0.70512003, -0.11909, 0.16256, -0.43259001, -0.050078001, 0.050232001, -0.1145, -0.041885, 0.47865999, 0.012767, 0.19642, 0.26196, -0.29425001, 0.089615002, -0.17736, -0.22448, 0.22623999, 0.16749001, 0.055769999, 0.14399, 0.2158, 0.33818999, 0.23458999, 0.15826, -0.28560001, 0.24199, 0.11018, 0.38163999, -0.29840001, -0.20169, 0.26949999, 0.11186, -0.21006, -0.042070001, 0.016507, -0.22866, -3.3882, 0.29203999, -0.088358, -0.014966, -0.25224999, -0.11503, 0.036336999, -0.14816999, 0.046220001, -0.073466003, -0.13866, 0.23612, 0.033882, 0.29495001, -0.61233997, 0.20288999, -0.42091, 0.37766999, 0.036260001, 0.21708, 0.12560999, -0.21682, -0.0037996999, -0.17791, -0.26431, 0.31678, -0.051229, 0.049268998, -0.12622, -0.10117, 0.017246, -0.021950001, -0.1982, 0.037250001, -0.16790999, -0.055459, 0.57669997, 0.059122998, 0.22931001, 0.064200997, 0.27423999, -0.37129, -0.091375001, -0.071341999, -0.037218001, -0.012668, -0.017976001, -0.42622, -0.10095, 0.044992, -0.090225004, 0.22915, 0.18610001, 0.36366001, -0.20676, -0.33037001, 0.47301999, 0.23379999, 0.079305999, 0.21083, 0.21013001, 0.15275, 0.080872998, -0.33013001, -0.17181, -0.07017, -0.041244, -0.46182001, 0.027903, 0.54657, -0.25894001, 0.39515001, 0.26144001, -0.54066002, 0.21199, -0.0094357003])
comma = np.array([0.18378, -0.12123, -0.11987, 0.015227, -0.19121, -0.066073999, -2.9876001, 0.80795002, 0.067337997, -0.13184001, -0.52740002, 0.44521001, 0.12982, -0.21822999, -0.4508, -0.22477999, -0.30766001, -0.11137, -0.162, -0.21294001, -0.46022001, -0.086593002, -0.24902, 0.46729001, -0.60229999, -0.44972, 0.43946001, 0.014738, 0.27498001, -0.078420997, 0.36008999, 0.12172, 0.4298, -0.055344999, 0.44949999, -0.74444002, -0.26701999, 0.16430999, -0.19335, 0.13468, 0.28870001, 0.23924001, -0.23579, -0.28972, 0.20149, 0.048135001, -0.18322, -0.15492, -0.19255, 0.40270999, 0.16051, 0.17721, 0.32556999, 0.011625, -0.42572001, 0.34204999, -0.45864999, -0.24860001, 0.034127999, 0.033059999, -0.057064999, 0.18136001, -0.43638, 0.00057089998, -0.11935, -0.21950001, 0.16429, -0.18119, -0.19145, -0.081671998, -0.29620001, 0.25803, 0.073848002, 0.54212999, -0.15404999, -0.49256, 0.091719002, 0.13328999, -0.052530002, -0.20518, 0.34575999, -1.0448999, 0.072779, -0.00034530001, -0.16926, 0.051019002, -0.14753, 0.23848, -0.40748999, -0.58278, -0.48695001, 0.25863001, -0.20531, -0.47749999, 0.40645, -0.038511999, -2.4030001, -0.12421, 0.63148999, 0.089419, 0.08557, -0.20757, -0.1617, -0.29506001, -0.13947999, 0.14202, -0.30138001, -0.15806, 0.52983999, 0.24229001, 0.075168997, 0.13792001, 0.90416002, -0.23647, 0.027788, 0.099914998, 0.45422, 0.60176003, 0.25044, 0.29142001, 0.040711999, -0.081210002, -0.43786001, -0.30149999, -0.17991, -0.52148998, 0.029446, -0.23051, 0.073954999, 0.34751001, 0.078060001, 0.19801, -0.32246, -0.13827001, 0.10076, 0.56601, 0.31924999, 0.09426, -0.045898002, 0.78329003, 0.19997001, 0.1619, 0.41578999, -0.31467, -0.036655001, -0.11687, -0.17941999, 0.16246, 0.42221001, 0.19588, -0.025057999, -0.018717, -0.17964999, 0.35635, 0.25852999, 0.13139001, 0.026784001, 0.017271001, -0.14781, 0.30598, -0.033227999, 0.15521, -0.50573999, 0.1295, 0.14602, -0.35552001, -0.43193999, -0.1029, 0.047359999, -0.57902998, -0.42488, 0.67163002, -0.11182, 0.29306, -0.0033312, 0.13090999, -0.086654998, 0.22618, 0.29357001, -0.30880001, -0.42704999, 0.32679999, 0.39254001, 0.17474, -0.19659001, 0.35664999, 0.38025001, 0.24257, -0.17021, 0.097295001, 0.45247999, -0.40588999, 0.27886, -0.33315, 0.37075999, 0.16742, -0.28582001, -0.051603999, -0.090346001, 0.095385, 0.26394999, -0.30008, -0.63243997, 0.076665998, 0.14102, 0.88612998, -0.053817, 0.26223001, -0.016005, -0.040608, 0.082135998, -0.081589997, -0.068911999, -0.62238997, -0.014757, -0.033402, 0.25847, -0.28878, -0.27142999, -0.23709001, -0.11285, 0.24828, 0.14511999, 0.3373, -4.1005001, -0.075260997, 0.32638001, 0.21444, 0.37972, 0.029262999, 0.24594, 0.42934999, 0.68689001, -0.58112001, 0.22939, -0.38889, 0.41683999, 0.066216998, 0.47900999, 0.27427, 0.41644999, -0.35492, -0.14413001, -0.010046, -0.42024001, -0.19382, 0.36155999, -0.13364001, -0.29853001, 0.47536999, -0.26989001, -0.083662003, -0.074100003, 0.21815, -0.30678001, -0.83499002, -0.11287, -0.32611999, 0.12375, 0.35341001, -0.32607001, 0.32853001, 0.060265999, -0.21991, 0.35670999, 0.29545999, -0.48159, -0.22347, 0.31036001, 0.22132, -0.20994, -0.085675001, -0.26172999, -0.10764, -0.14802, 0.17573, -0.17804, -0.21765, 0.3073, -0.44589999, 0.039129999, -0.22065, 0.22139999, 0.32727, -0.40378001, 0.33021, -0.13942, -0.41003001, -0.17526001, 0.21852, 0.13615, 0.10999, -0.33474001, -0.046108998, 0.1078, -0.035657, -0.012921, -0.039037999, 0.18274, 0.14654])
first_data = np.array([-0.10974872, 0.037625644, -0.084976442, -0.047660731, 0.085314184, 0.078813098, -3.273432, 0.25345027, 0.051668897, -0.58701181, 0.28418729, 0.15717727, 0.11167406, -0.20312573, -0.17073238, -0.060610089, -0.069798432, -0.1256019, 0.02125809, 0.098465092, 0.16126347, 0.01063936, 0.091032907, 0.18358026, 0.073181048, -0.11374073, -0.087250821, -0.22090043, -0.057654724, -0.076624088, -0.10258137, 0.015447184, 0.18705869, 0.070748895, -0.10415955, 0.08780691, 0.12742336, -0.070803456, 0.12418309, -0.073622353, 0.021656273, 0.1901651, -0.14402114, -0.18927065, -0.0090129161, 0.081275098, 0.024802273, -0.11792881, -0.016291911, 0.024433363, -0.099074006, -0.068325721, -0.097144172, 0.016342273, 0.083744369, -0.16438426, 0.040059395, 0.055647004, 0.18537465, 0.035565998, 0.03616685, -0.16052629, 0.12714228, 0.005990474, 0.040781546, 0.1546791, -0.030567819, 0.080147997, 0.0088854656, -0.16850996, -0.11115146, 0.15598637, 0.025624096, 0.0077750902, -0.017262543, -0.2132491, 0.24513064, 0.057064869, 0.023979366, 0.11817663, 0.027577706, -0.60866278, -0.16747671, -0.046188179, -0.105493, -0.032166634, -0.11850727, 0.02995109, -0.014456887, -0.030454041, 0.11530101, -0.10403728, -0.10518742, 0.092550181, 0.050916635, -0.241197, -1.6692173, -0.024015307, 0.10750155, 0.16796038, -0.13808519, 0.11869964, -0.077704184, -0.048279729, -0.15865381, -0.020840693, -0.039821092, -0.35068589, 8.781525e-05, 0.001829276, -0.03816691, -0.098226659, -0.12422854, 0.083839722, -0.17595173, 0.16005883, 0.29587963, -0.14986689, 0.088956997, 0.11557946, -0.070624821, -0.075243644, 0.087942094, -0.031316031, 0.0524619, -0.20974539, 0.10225518, -0.071169265, -0.036148001, 0.035048064, -0.22765614, 0.072583742, 0.04255591, -0.073108368, -0.068275362, 0.051201191, 0.09777382, 0.01618327, 0.052368272, 0.55681634, 0.21886091, -0.0095886355, 0.11900394, -0.024176361, 0.0048634545, -0.26624447, 0.17964457, -0.045330547, 0.20256172, 0.0097496789, 0.089637995, -0.08556623, 0.059150089, -0.17483918, 0.13023537, -0.023737907, -0.11014804, 0.061376452, -0.052483462, 0.10754354, 0.10079221, 0.0027729978, -0.22834483, 0.034409408, -0.03773528, 0.028101813, -0.056789637, -0.12804548, 0.095311359, -0.047766373, -0.0095169097, -0.036243096, -0.09934663, -0.08010745, -0.12684862, 0.16092917, 0.028561972, -0.0088372733, -0.0076575428, -0.16672298, 0.10041917, 0.21157672, 0.082259998, 0.02162854, -0.051445208, -0.18189582, 0.068763427, 0.093136184, 0.19526991, 0.054629311, 0.10514446, -0.12583245, 0.13162018, -0.063150696, 0.018314181, -0.003260538, 0.072106294, 0.065206267, 0.095836729, -0.074389905, 0.082492918, -0.14763655, -0.15195474, -0.0039215563, 0.00642162, 0.15423809, -0.14701492, -0.040211633, 0.24132672, -0.075294547, 0.055631366, -0.206075, 0.1626953, -0.10290847, -0.017531095, -0.077754095, -0.034012273, -0.035827823, -0.0050435443, -0.1206825, 0.060781911, -0.183596, -0.14618967, 0.012038544, -2.2940364, -0.012405452, -0.026705861, 0.00196736, 0.10820363, -0.14092447, 0.06243746, 0.10743809, 0.098148629, -0.019588815, -0.22887836, 0.00061963632, -0.011909729, 0.077108733, -0.027317367, 0.035651729, -0.085503586, -0.029489366, -0.23672675, 0.28352073, -0.048629358, 0.016084468, -0.11548565, 0.1801824, 0.09046708, 0.026665092, -0.065884694, 0.063704573, 0.027784998, 0.062916994, 0.25671253, -0.16826801, 0.071896911, -0.070956551, -0.073734544, 0.10302675, 0.00013363361, -0.08252912, 0.12834843, -0.098448388, -0.060259733, 0.038886908, -0.22094183, 0.0067698453, 0.12068355, -0.082720436, 0.166706, -0.16707362, -0.18674301, 0.097038537, -0.060636356, -0.058469817, 0.040454391, 0.15812901, 0.14724344, 0.056362633, 0.61374187, -0.07254985, 0.0099259056, 0.048752367, 0.11264255, -0.041378822, 0.040786996, 0.16385324, -0.067051634, -0.2118782, -0.063515455, -0.071630545, -0.24930356, -0.030746095, 0.047875356, -0.10911128, -0.17182106, -0.16525608, 0.0081145559, -0.012611181])

def sentence(toUse):
    text = toUse.split()
    sentence_vector = np.zeros(300, dtype=np.float32)
    for word in text:
        try:
            word = word.strip(punctuation)
            sentence_vector += vectors[index_map[word]]
        except:
            pass
    # average found here
    sentence_vector /= len(text)
    return list(sentence_vector)

while True:
    testInput = np.array(sentence(input("Input to classify sentence: ")))

    test = np.reshape(testInput,(1,-1))
    #start = time.time()
    #this_is = loaded_model.predict(test)[0]
    #end = time.time()
    #print("time: ", end - start)
    if loaded_model.predict(test)[0] == 1:
        print("Bullying detected!")
    else:
        print("No bullying detected")