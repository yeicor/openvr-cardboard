package dev.openvrcardboard.phone

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * Verifies the Kotlin fallback ladder against the shared protocol files used
 * by the desktop Python tests.
 */
class CameraModesTest {
    private fun protocolFile(name: String): File {
        val golden = System.getProperty("ovrc.goldenVectors")
            ?: "../../protocol/golden_vectors.txt"
        val parent = File(golden).absoluteFile.parentFile ?: File(".")
        return parent.resolve(name)
    }

    private fun mode(values: List<Int>): CameraModeRequest {
        assertEquals(3, values.size)
        return CameraModeRequest(values[0], values[1], values[2])
    }

    @Test
    fun defaultLadderMatchesSharedSpec() {
        val expected = protocolFile("camera_modes.txt")
            .readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
            .map { line ->
                mode(line.split(Regex("\\s+")).map { it.toInt() })
            }
        assertEquals(expected, CameraModes.DEFAULT_MODES)
    }

    @Test
    fun fallbackAttemptsMatchSharedCases() {
        val cases = protocolFile("camera_mode_attempts.txt")
            .readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
        assertEquals(15, cases.size)
        for (line in cases) {
            val (targetText, attemptsText) = line.split("|", limit = 2)
            val target = mode(
                targetText.trim().split(Regex("\\s+")).map { it.toInt() },
            )
            val expected = attemptsText
                .split(",")
                .map { part ->
                    mode(part.trim().split(Regex("\\s+")).map { it.toInt() })
                }
            assertEquals("target $target", expected, CameraModes.attempts(target))
        }
    }
}
