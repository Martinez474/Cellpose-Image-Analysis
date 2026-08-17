import ij.IJ;
import ij.ImagePlus;
import ij.gui.PolygonRoi;
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
import java.util.concurrent.TimeUnit;

/** ImageJ entry point for running Cellpose and displaying prediction polygons. */
public class Run_AI_Detection implements PlugIn {
    private static final Path PROJECT_DIRECTORY = Paths.get(
            "/home/mars/Documents/school/thesis/Cellpose-Image-Analysis"
    );
    private static final Path PYTHON_PATH = PROJECT_DIRECTORY.resolve(".venv/bin/python");
    private static final Path PREDICT_SCRIPT = PROJECT_DIRECTORY.resolve("predict.py");
    private static final Path PREDICTIONS_PATH = PROJECT_DIRECTORY.resolve("predictions.json");
    private static final long PREDICTION_TIMEOUT_MINUTES = 30;

    @Override
    public void run(String arg) {
        ImagePlus image = IJ.getImage();

        if (image == null) {
            IJ.showMessage("Run AI Detection", "No image is currently open.");
            return;
        }

        try {
            Path imagePath = getImagePath(image);
            validateAiFiles();

            IJ.showStatus("Running Cellpose on " + imagePath.getFileName() + "...");
            String predictionOutput = runPrediction(imagePath);
            if (!predictionOutput.isEmpty()) {
                IJ.log("Cellpose output:\n" + predictionOutput);
            }

            PredictionFile predictionFile = JsonLoader.load(PREDICTIONS_PATH.toString());
            List<PolygonRoi> rois = RoiCreator.createRois(predictionFile.predictions);

            if (rois.isEmpty()) {
                IJ.showMessage("Run AI Detection", "No valid polygons found in JSON.");
                return;
            }

            RoiManager roiManager = RoiManager.getInstance();
            if (roiManager == null) {
                roiManager = new RoiManager();
            }

            for (PolygonRoi roi : rois) {
                roiManager.addRoi(roi);
            }

            roiManager.setVisible(true);
            roiManager.runCommand(image, "Show All");
            image.updateAndDraw();
            IJ.showStatus("Cellpose finished: " + rois.size() + " ROIs added.");

            IJ.showMessage(
                    "Run AI Detection",
                    "Added " + rois.size() + " AI prediction ROIs."
            );
        } catch (Exception e) {
            String message = e.getMessage() == null ? e.toString() : e.getMessage();
            IJ.showMessage("Run AI Detection Error", message);
            IJ.log(e.toString());
        }
    }

    private Path getImagePath(ImagePlus image) throws Exception {
        FileInfo fileInfo = image.getOriginalFileInfo();

        if (fileInfo == null || fileInfo.directory == null || fileInfo.fileName == null) {
            throw new Exception(
                    "The current image has not been saved. Save it as an image file, "
                    + "reopen it, and run the plugin again."
            );
        }

        Path imagePath = Paths.get(fileInfo.directory, fileInfo.fileName).toAbsolutePath();
        if (!Files.isRegularFile(imagePath)) {
            throw new Exception("The current image file does not exist: " + imagePath);
        }
        return imagePath;
    }

    private void validateAiFiles() throws Exception {
        if (!Files.isExecutable(PYTHON_PATH)) {
            throw new Exception("Python environment was not found: " + PYTHON_PATH);
        }
        if (!Files.isRegularFile(PREDICT_SCRIPT)) {
            throw new Exception("Prediction script was not found: " + PREDICT_SCRIPT);
        }
    }

    private String runPrediction(Path imagePath) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder(
                PYTHON_PATH.toString(),
                PREDICT_SCRIPT.toString(),
                imagePath.toString(),
                PREDICTIONS_PATH.toString()
        );
        processBuilder.directory(new File(PROJECT_DIRECTORY.toString()));
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();
        StringBuilder output = new StringBuilder();

        Thread outputReader = new Thread(new Runnable() {
            @Override
            public void run() {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        output.append(line).append(System.lineSeparator());
                        IJ.log("Cellpose: " + line);
                    }
                } catch (Exception e) {
                    output.append("Unable to read Cellpose output: ")
                            .append(e.getMessage())
                            .append(System.lineSeparator());
                }
            }
        }, "cellpose-output-reader");
        outputReader.start();

        boolean finished = process.waitFor(PREDICTION_TIMEOUT_MINUTES, TimeUnit.MINUTES);
        if (!finished) {
            process.destroyForcibly();
            outputReader.join();
            throw new Exception(
                    "Cellpose did not finish within " + PREDICTION_TIMEOUT_MINUTES
                    + " minutes.\n" + output.toString().trim()
            );
        }

        outputReader.join();
        if (process.exitValue() != 0) {
            throw new Exception(
                    "Cellpose exited with code " + process.exitValue()
                    + ".\n" + output.toString().trim()
            );
        }
        if (!Files.isRegularFile(PREDICTIONS_PATH)) {
            throw new Exception("Cellpose finished without creating " + PREDICTIONS_PATH);
        }

        return output.toString().trim();
    }
}
