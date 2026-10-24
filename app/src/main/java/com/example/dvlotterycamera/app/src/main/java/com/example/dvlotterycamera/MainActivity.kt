package com.example.dvlotterycamera

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs

data class ValidationResult(
    val isValid: Boolean,
    val message: String
)

class FaceAnalyzer(
    private val context: Context,
    private val onValidationUpdate: (ValidationResult) -> Unit,
    private val onTriggerAutoCapture: () -> Unit
) : ImageAnalysis.Analyzer {

    private var faceLandmarker: FaceLandmarker? = null
    private var validStateStartTime: Long = 0L
    private var isCapturing: Boolean = false

    init {
        setupLandmarker()
    }

    private fun setupLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFaceTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(context, options)
    }

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap().rotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        val mpImage = BitmapImageBuilder(bitmap).build()

        val result = faceLandmarker?.detect(mpImage)

        if (result != null && result.faceLandmarks().isNotEmpty()) {
            val landmarks = result.faceLandmarks()[0]
            val validation = validateFace(landmarks)

            if (validation.isValid) {
                if (validStateStartTime == 0L) {
                    validStateStartTime = SystemClock.elapsedRealtime()
                } else if (SystemClock.elapsedRealtime() - validStateStartTime >= 2000L && !isCapturing) {
                    isCapturing = true
                    onTriggerAutoCapture()
                }
            } else {
                validStateStartTime = 0L
            }

            onValidationUpdate(validation)
        } else {
            validStateStartTime = 0L
            onValidationUpdate(ValidationResult(false, "وجه كاميرا الهاتف نحو وجهك"))
        }

        imageProxy.close()
    }

    private fun validateFace(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): ValidationResult {
        val noseTip = landmarks[1]
        val leftCheek = landmarks[234]
        val rightCheek = landmarks[454]
        
        val leftDistance = abs(noseTip.x() - leftCheek.x())
        val rightDistance = abs(noseTip.x() - rightCheek.x())
        val yawRatio = leftDistance / rightDistance

        if (yawRatio < 0.6f || yawRatio > 1.6f) {
            return ValidationResult(false, "اضبط وجهك ليكون مستقيماً تماماً")
        }

        val leftEyeTop = landmarks[159]
        val leftEyeBottom = landmarks[145]
        val rightEyeTop = landmarks[386]
        val rightEyeBottom = landmarks[374]

        if (abs(leftEyeTop.y() - leftEyeBottom.y()) < 0.015f || abs(rightEyeTop.y() - rightEyeBottom.y()) < 0.015f) {
            return ValidationResult(false, "افتح عينيك جيداً وتطلع للكاميرا")
        }

        val mouthLeft = landmarks[61]
        val mouthRight = landmarks[291]
        val mouthTop = landmarks[13]
        val mouthBottom = landmarks[14]

        if ((abs(mouthTop.y() - mouthBottom.y()) / abs(mouthLeft.x() - mouthRight.x())) > 0.35f) {
            return ValidationResult(false, "حافظ على تعبير وجه محايد بدون ابتسامة")
        }

        val chin = landmarks[152]
        val topHead = landmarks[10]
        val faceHeightRatio = abs(chin.y() - topHead.y())

        if (faceHeightRatio < 0.45f) return ValidationResult(false, "اقترب قليلاً من الكاميرا")
        if (faceHeightRatio > 0.72f) return ValidationResult(false, "ابتعد قليلاً عن الكاميرا")

        return ValidationResult(true, "ثبّت يدك.. جاري الالتقاط تلقائياً")
    }

    fun resetCaptureState() {
        isCapturing = false
        validStateStartTime = 0L
    }

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }
}

object ImageProcessor {
    fun processDVLotteryImage(inputBitmap: Bitmap): Bitmap {
        val bgRemovedBitmap = removeBackground(inputBitmap)
        val squareBitmap = cropToSquare(bgRemovedBitmap)
        return Bitmap.createScaledBitmap(squareBitmap, 600, 600, true)
    }

    private fun removeBackground(bitmap: Bitmap): Bitmap {
        val options = SelfieSegmenterOptions.Builder()
            .setSegmenterMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .build()
        val segmenter = Segmentation.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)

        val task = segmenter.process(image)
        val segmentationMask = Tasks.await(task)

        val maskBuffer: ByteBuffer = segmentationMask.buffer
        val maskWidth = segmentationMask.width
        val maskHeight = segmentationMask.height

        val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        canvas.drawColor(Color.WHITE)

        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val scaleX = maskWidth.toFloat() / bitmap.width
        val scaleY = maskHeight.toFloat() / bitmap.height

        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val maskX = (x * scaleX).toInt().coerceIn(0, maskWidth - 1)
                val maskY = (y * scaleY).toInt().coerceIn(0, maskHeight - 1)
                val confidence = maskBuffer.getFloat((maskY * maskWidth + maskX) * 4)

                if (confidence > 0.45f) {
                    outputBitmap.setPixel(x, y, pixels[y * bitmap.width + x])
                }
            }
        }
        return outputBitmap
    }

    private fun cropToSquare(bitmap: Bitmap): Bitmap {
        val newSize = if (bitmap.width < bitmap.height) bitmap.width else bitmap.height
        return Bitmap.createBitmap(bitmap, (bitmap.width - newSize) / 2, (bitmap.height - newSize) / 2, newSize, newSize)
    }
}

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = null
    private var lastSpokenMessage = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(this, this)

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = ComposeColor.Black) {
                    DVCameraScreen(onSpeak = { speakOut(it) })
                }
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale("ar")
        }
    }

    private fun speakOut(text: String) {
        if (text != lastSpokenMessage) {
            lastSpokenMessage = text
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun DVCameraScreen(onSpeak: (String) -> Unit) {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    LaunchedEffect(Unit) {
        if (!cameraPermissionState.status.isGranted) {
            cameraPermissionState.launchPermissionRequest()
        }
    }

    if (cameraPermissionState.status.isGranted) {
        CameraContent(onSpeak = onSpeak)
    } else {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("يرجى إعطاء إذن الكاميرا لتشغيل التطبيق", color = ComposeColor.White, fontSize = 18.sp)
        }
    }
}

@Composable
fun CameraContent(onSpeak: (String) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    var statusMessage by remember { mutableStateOf("جاري التعرف على الوجه...") }
    var isGreenFrame by remember { mutableStateOf(false) }
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    val imageCapture = remember { ImageCapture.Builder().build() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val analyzer = remember {
        FaceAnalyzer(
            context = context,
            onValidationUpdate = { validation ->
                statusMessage = validation.message
                isGreenFrame = validation.isValid
                onSpeak(validation.message)
            },
            onTriggerAutoCapture = {
                imageCapture.takePicture(
                    cameraExecutor,
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            isProcessing = true
                            val bitmap = image.toBitmap()
                            image.close()

                            scope.launch(Dispatchers.Default) {
                                val result = ImageProcessor.processDVLotteryImage(bitmap)
                                processedBitmap = result
                                isProcessing = false
                            }
                        }
                    }
                )
            }
        )
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (processedBitmap == null) {
            AndroidView(
                factory = { ctx ->
                    val previewView = PreviewView(ctx)
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            .also {
                                it.setAnalyzer(cameraExecutor, analyzer)
                            }

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_FRONT_CAMERA,
                                preview,
                                imageCapture,
                                imageAnalysis
                            )
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }, ContextCompat.getMainExecutor(ctx))

                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )

            Canvas(modifier = Modifier.fillMaxSize()) {
                val strokeColor = if (isGreenFrame) ComposeColor.Green else ComposeColor.Red
                val ovalWidth = size.width * 0.7f
                val ovalHeight = size.height * 0.5f

                drawOval(
                    color = strokeColor,
                    topLeft = androidx.compose.ui.geometry.Offset((size.width - ovalWidth) / 2, (size.height - ovalHeight) / 2.3f),
                    size = androidx.compose.ui.geometry.Size(ovalWidth, ovalHeight),
                    style = Stroke(width = 8.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(20f, 20f), 0f))
                )
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .align(Alignment.BottomCenter)
                    .padding(24.dp)
                    .background(ComposeColor.Black.copy(alpha = 0.75f), RoundedCornerShape(16.dp))
                    .padding(16.dp)
            ) {
                Text(
                    text = statusMessage,
                    color = if (isGreenFrame) ComposeColor.Green else ComposeColor.Yellow,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            if (isProcessing) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(ComposeColor.Black.copy(alpha = 0.6f)),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = ComposeColor.White)
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("جاري معالجة الصورة حسب معايير اللوتري...", color = ComposeColor.White, fontSize = 16.sp)
                    }
                }
            }
        } else {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(ComposeColor.Black)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    "الصورة النهائية المطابقة للشروط (600x600)",
                    color = ComposeColor.Green,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                Image(
                    bitmap = processedBitmap!!.asImageBitmap(),
                    contentDescription = "Final DV Image",
                    modifier = Modifier
                        .size(300.dp)
                        .border(2.dp, ComposeColor.White)
                )

                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = {
                        processedBitmap = null
                        analyzer.resetCaptureState()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = ComposeColor.White)
                ) {
                    Text("إعادة الالتقاط", color = ComposeColor.Black, fontSize = 16.sp)
                }
            }
        }
    }
}
