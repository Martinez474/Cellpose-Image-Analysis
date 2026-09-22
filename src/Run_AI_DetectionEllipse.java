import ij.IJ;
import ij.ImagePlus;
import ij.gui.PolygonRoi;
import ij.io.DirectoryChooser;
import ij.io.FileInfo;
import ij.plugin.PlugIn;
import ij.plugin.frame.RoiManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** ImageJ entry point for the experimental OpenCV ellipse detector. */
public class Run_AI_DetectionEllipse implements PlugIn {
    private static ProjectSettings settings;
    private static final long PREDICTION_TIMEOUT_MINUTES = 30;

    @Override
    public void run(String arg) {
        ImagePlus image = IJ.getImage();
        if (image == null) {
            IJ.showMessage("Run Ellipse Detection", "No image is currently open.");
            return;
        }
        try {
            ensureSettings();
            Path imagePath = getImagePath(image);
            Path script = settings.projectDirectory.resolve("predict_ellipse.py");
            Path output = settings.projectDirectory.resolve("ellipse_predictions.json");
            if (!Files.isRegularFile(script)) {
                throw new Exception("Ellipse prediction script was not found: " + script);
            }
            if (!Files.isExecutable(settings.pythonPath)) {
                throw new Exception("Python environment was not found: " + settings.pythonPath);
            }

            IJ.showStatus("Detecting ellipses in " + imagePath.getFileName() + "...");
            String outputLog = runPrediction(imagePath, script, output);
            if (!outputLog.isEmpty()) IJ.log("Ellipse detector output:\n" + outputLog);

            PredictionFile predictionFile = JsonLoader.load(output.toString());
            List<PolygonRoi> rois = RoiCreator.createRois(predictionFile.predictions);
            if (rois.isEmpty()) {
                IJ.showMessage("Run Ellipse Detection", "No ellipse polygons found.");
                return;
            }
            RoiManager roiManager = RoiManager.getInstance();
            if (roiManager == null) roiManager = new RoiManager();
            for (PolygonRoi roi : rois) roiManager.addRoi(roi);
            roiManager.setVisible(true);
            roiManager.runCommand(image, "Show All");
            image.updateAndDraw();
            IJ.showStatus("Ellipse detection finished: " + rois.size() + " ROIs added.");
            IJ.showMessage("Run Ellipse Detection", "Added " + rois.size() + " ellipse ROIs.");
        } catch (Exception error) {
            String message = error.getMessage() == null ? error.toString() : error.getMessage();
            IJ.showMessage("Run Ellipse Detection Error", message);
            IJ.log(error.toString());
        }
    }

    private void ensureSettings() throws Exception {
        if (settings != null) return;
        try {
            settings = ProjectSettings.load();
        } catch (IllegalStateException automaticDetectionError) {
            DirectoryChooser chooser = new DirectoryChooser("Select Cellpose project folder");
            String selected = chooser.getDirectory();
            if (selected == null) throw automaticDetectionError;
            settings = ProjectSettings.fromProject(Paths.get(selected));
        }
    }

    private Path getImagePath(ImagePlus image) throws Exception {
        FileInfo info = image.getOriginalFileInfo();
        if (info == null || info.directory == null || info.fileName == null) {
            throw new Exception("Save and reopen the current image before running the plugin.");
        }
        Path path = Paths.get(info.directory, info.fileName).toAbsolutePath();
        if (!Files.isRegularFile(path)) throw new Exception("Image file does not exist: " + path);
        return path;
    }

    private String runPrediction(Path image, Path script, Path output) throws Exception {
        ProcessBuilder builder = new ProcessBuilder(
                settings.pythonPath.toString(), script.toString(), image.toString(), output.toString());
        builder.directory(new File(settings.projectDirectory.toString()));
        builder.redirectErrorStream(true);
        Map<String, String> environment = builder.environment();
        environment.put("__NV_PRIME_RENDER_OFFLOAD", "1");
        environment.put("__GLX_VENDOR_LIBRARY_NAME", "nvidia");
        Process process = builder.start();
        StringBuilder log = new StringBuilder();
        Thread readerThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log.append(line).append(System.lineSeparator());
                    IJ.log("Ellipse detector: " + line);
                }
            } catch (Exception error) {
                log.append("Unable to read ellipse detector output: ").append(error.getMessage());
            }
        }, "ellipse-detector-output-reader");
        readerThread.start();
        if (!process.waitFor(PREDICTION_TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
            process.destroyForcibly();
            readerThread.join();
            throw new Exception("Ellipse detector timed out.\n" + log.toString().trim());
        }
        readerThread.join();
        if (process.exitValue() != 0) {
            throw new Exception("Ellipse detector exited with code " + process.exitValue()
                    + ".\n" + log.toString().trim());
        }
        if (!Files.isRegularFile(output)) {
            throw new Exception("Ellipse detector did not create " + output);
        }
        return log.toString().trim();
    }
}
