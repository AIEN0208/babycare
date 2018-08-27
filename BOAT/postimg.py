import requests


def photo_send(imagename):   # send photo to website server
	url = 'http://13.229.127.168/api/paint/'
	datas = {'UserId': 1, 'ImageName' : imagename}
	files = {'Image': open(imagename, 'rb')}
	response = requests.post(url, datas, files = files)
	# print(response.text)

if __name__ == '__main__':
	photo_send('artwork.jpg')
