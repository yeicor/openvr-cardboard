import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
}

android {
    namespace = "dev.openvrcardboard.phone"
    // AGP 9 resolves the versioned SDK platform (platforms;android-37.0).
    compileSdk {
        version = release(37)
    }
    buildToolsVersion = "37.0.0"

    defaultConfig {
        applicationId = "dev.openvrcardboard.phone"
        minSdk = 26
        targetSdk = 37
        versionCode = 3
        versionName = "1.1.0"
    }

    // Release APKs must always be signed with the same key, otherwise Android
    // refuses to install an update over the previous version. CI decodes the
    // keystore from GitHub secrets into a temporary file and points
    // OVRC_ANDROID_KEYSTORE at it; the keystore and passwords never enter the
    // repository. A local release build without these variables is unsigned.
    signingConfigs {
        val keystorePath = System.getenv("OVRC_ANDROID_KEYSTORE")
        val keystorePassword = System.getenv("OVRC_ANDROID_KEYSTORE_PASSWORD")
        val keyAliasName = System.getenv("OVRC_ANDROID_KEY_ALIAS")
        val keyPasswordValue = System.getenv("OVRC_ANDROID_KEY_PASSWORD")
        if (
            !keystorePath.isNullOrBlank() &&
            !keystorePassword.isNullOrBlank() &&
            !keyAliasName.isNullOrBlank() &&
            !keyPasswordValue.isNullOrBlank()
        ) {
            create("release") {
                storeFile = file(keystorePath)
                storePassword = keystorePassword
                keyAlias = keyAliasName
                keyPassword = keyPasswordValue
            }
        }
    }

    buildTypes {
        release {
            signingConfigs.findByName("release")?.let { signingConfig = it }
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    testOptions {
        unitTests.all {
            it.systemProperty(
                "ovrc.goldenVectors",
                rootProject.file("../protocol/golden_vectors.txt").absolutePath,
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// AGP 9 compiles Kotlin with built-in Kotlin support, so the old
// org.jetbrains.kotlin.android plugin and android.kotlinOptions block are gone.
kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.19.1")
    implementation("androidx.appcompat:appcompat:1.8.0")
    implementation("androidx.activity:activity-ktx:1.13.0")
    implementation("com.google.android.material:material:1.14.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    implementation("com.squareup.okhttp3:okhttp:5.5.1")

    testImplementation("junit:junit:4.13.2")
}
