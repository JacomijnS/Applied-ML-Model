<template>
  <q-page class="flex flex-center column">
    <h3>Welcome!</h3>
    <p style="font-size: 20px;">This is the frontend of our Applied Machine Learning project. </p>
    <p style="font-size: 20px;"> Our model is suppose to detect fractures when given a picture. Give this a try in the left tab or click on the button! </p>
    <q-btn color="primary" label="Go to our model" @click="goToModelPage" />
  </q-page>
</template>

<script setup>
import { useRouter } from 'vue-router'
const router = useRouter()

function goToModelPage() {
  router.push('/model')
}
</script>
