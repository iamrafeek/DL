import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle 

###############################################################################

image_dir="cropped"
lionel_messi=os.listdir(image_dir+ '/lionel_messi')
maria_sharapova=os.listdir(image_dir+ '/maria_sharapova')
roger_federer=os.listdir(image_dir+ '/roger_federer')
serena_williams=os.listdir(image_dir+ '/serena_williams')
virat_kohli=os.listdir(image_dir+ '/virat_kohli')

dataset=[]
label=[]
img_siz=(128,128)


###############################################################################

for i , image_name in tqdm(enumerate(lionel_messi),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_sharapova),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i , image_name in tqdm(enumerate(roger_federer),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)
        
        
for i ,image_name in tqdm(enumerate(serena_williams),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)


for i , image_name in tqdm(enumerate(virat_kohli),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)
        
###############################################################################
        
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")



###############################################################################

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)



model= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

model.summary()

print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=50,batch_size =128,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")


# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('classification_accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('classification_loss_plot.png')


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred = model.predict(x_test)
predicted_labels = np.argmax(y_pred, axis=1)
print('classification Report\n',classification_report(y_test,predicted_labels))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")

def make_prediction(img_path, model):
    img = cv2.imread(img_path)
    img = Image.fromarray(img, 'RGB')  # Assuming images are in RGB format
    img = img.resize((128, 128))
    img = np.array(img)
    
    # Normalize the image
    img = img / 255.0
    
    # Expand dimensions to match the input shape expected by the model
    input_img = np.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(input_img)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    
    # Map class index to class label (you may need to adjust this based on your classes)
    class_labels = {0: 'lionel_messi', 1: 'maria_sharapova', 2: 'roger_federer', 3: 'serena_williams', 4: 'virat_kohli'}
    predicted_label = class_labels[predicted_class]
    
    print(f"Predicted class: {predicted_class}, Predicted label: {predicted_label}, Confidence: {predictions[0][predicted_class]}")

# Example usage
image_path_to_predict = "cropped/virat_kohli/virat_kohli2.png"
make_prediction(image_path_to_predict, model)

        




# Save the entire model to a HDF5 file
model.save('img_classification_model.h5')

with open('img_classification_model.json', 'wb') as model_file:
    pickle.dump(model,model_file)

