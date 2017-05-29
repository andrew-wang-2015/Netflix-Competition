import numpy as np
u_pred_qual = np.loadtxt('final_predictions_all_user.txt')
new_pred = np.zeros(len(u_pred_qual))
for i, pred in np.ndenumerate(u_pred_qual):
	'''
	if (i[0] % 1000) == 0:
		bar.update(i[0] % bar.max_value)
	'''
	curr_pred = pred
	#user = qual_I[i]
	#movie = qual_J[i]
	#print(user)
	'''
	if users_rated[user] == 0:
		print ("No ratings for this user: ", user)
		curr_pred = movie_avg[movie]
	
	if curr_pred > 5:
		curr_pred = 5
	if curr_pred < 1:
		curr_pred = 1

	curr_pred += movie_avg[movie] - pred_movie_avg[movie]

	'''
	decimal = curr_pred - int(curr_pred)
	if decimal <= 0.1:
		curr_pred = int(curr_pred)
	if decimal >= 0.9:
		curr_pred = int(curr_pred) + 1
	
	new_pred[i] = curr_pred
	#pred.write('%.3f\n' % curr_pred)
#bar.finish()

np.savetxt('final_predictions_all_user_rounded.txt', new_pred, fmt='%.3f')