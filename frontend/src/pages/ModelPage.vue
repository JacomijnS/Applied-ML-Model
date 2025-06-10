<template>
  <q-page>
    <div class="row" style="width: 100%; height: 100%">
      <div
        style="width: 35%; height: 92vh; border-right: 1px dashed gray"
        class="column items-center"
      >
        <q-uploader
          url="http://localhost:8000/predict"
          field-name="file"
          color="primary"
          text-color="white"
          label="Upload an image for prediction"
          accept=".jpg,.jpeg,.png"
          :auto-upload="true"
          @uploaded="onUploaded"
          @failed="onFailed"
          style="max-width: none; width: 90%; min-height: 300px; max-height: 80vh; margin-top: 5vh"
          bordered=""
          @added="onAdded"
        />
        <div v-if="error" class="q-mt-md text-negative">
          {{ error }}
        </div>
      </div>

      <div v-if="error" class="q-mt-md text-negative">
        {{ error }}
      </div>
      <div
        style="width: 65%; height: 92vh; flex-wrap: nowrap; overflow-y: auto"
        class="column items-center"
      >
        <q-card
          v-for="(prediction, idx) in predictions"
          :key="idx"
          class="q-my-sm"
          style="width: 80%"
        >
          <q-card-section>
            <div class="row items-center justify-around">
              <div>
                <div class="text-h6">Prediction Result</div>
                <div><b>File:</b> {{ prediction.filename }}</div>
                <div><b>Class:</b> {{ prediction.name }}</div>
                <div><b>Confidence:</b> {{ (prediction.confidence * 100).toFixed(2) }}%</div>
                <div v-if="prediction.bbox">
                  <b>Bounding Box:</b>
                  <pre class="q-ma-none q-pa-none">{{ prediction.bbox }}</pre>
                </div>
                <div v-else>
                  <b>No bounding box found.</b>
                </div>
              </div>
              <div style="width: 300px">
                <q-img
                  :src="prediction.imgUrl"
                  style="
                    width: 300px;
                    height: 300px;
                    object-fit: cover;
                    border-radius: 8px;
                    margin-right: 1em;
                  "
                  v-if="prediction.imgUrl"
                />
                <div style="width: 300px" class="row reverse q-mt-sm">
                  <q-btn
                    v-if="prediction.imgUrl"
                    color="secondary"
                    icon="zoom_in"
                    class="q-ml-md"
                    @click="showImageDialog(prediction.imgUrl)"
                    round
                  />
                </div>
              </div>
            </div>
          </q-card-section>
        </q-card>
      </div>
    </div>
  </q-page>
  <q-dialog v-model="imageDialogOpen">
    <q-card style="width: 90vw; max-height: 100%">
      <q-card-section>
        <q-img :src="dialogImgUrl" style="max-width: 100%; max-height: 80vh" />
      </q-card-section>
      <q-card-actions align="right">
        <q-btn flat label="Close" v-close-popup />
      </q-card-actions>
    </q-card>
  </q-dialog>
</template>

<script setup>
import { ref } from 'vue'

const predictions = ref([])
const error = ref(null)
const imgUrlMap = ref({})

const imageDialogOpen = ref(false)
const dialogImgUrl = ref('')

// generate preview URLs for each picture
function onAdded(event) {
  let files = event.files || event 
  if (!files || !files.forEach) {
    return
  }
  files.forEach((file) => {
    if (imgUrlMap.value[file.name]) {
      URL.revokeObjectURL(imgUrlMap.value[file.name])
    }
    imgUrlMap.value[file.name] = URL.createObjectURL(file)
    console.log('[onAdded]', file.name, imgUrlMap.value[file.name])
  })
}

// handle when files are uplaoded
function onUploaded({ xhr }) {
  try {
    const response = JSON.parse(xhr.response)
    response.imgUrl = imgUrlMap.value[response.filename] || null
    predictions.value.unshift(response)
    error.value = null
  } catch (err) {
    error.value = 'Invalid server response'
    console.log(err)
  }
}

// error handling
function onFailed() {
  error.value = 'Upload failed. Please try again.'
}

function showImageDialog(imgUrl) {
  dialogImgUrl.value = imgUrl
  imageDialogOpen.value = true
}
</script>
