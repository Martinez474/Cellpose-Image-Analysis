import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.Reader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/** Finds the project and applies optional cellpose-config.json overrides. */
final class ProjectSettings {
    final Path projectDirectory;
    final Path pythonPath;
    final Path predictScript;
    final Path predictionsPath;

    private ProjectSettings(Path projectDirectory, Path pythonPath,
                            Path predictScript, Path predictionsPath) {
        this.projectDirectory = projectDirectory;
        this.pythonPath = pythonPath;
        this.predictScript = predictScript;
        this.predictionsPath = predictionsPath;
    }

    static ProjectSettings load() {
        List<Path> starts = new ArrayList<Path>();
        starts.add(Paths.get(System.getProperty("user.dir")));
        try {
            URI location = Run_AI_Detection.class.getProtectionDomain()
                    .getCodeSource().getLocation().toURI();
            starts.add(Paths.get(location));
        } catch (Exception ignored) {
            // The working directory is still a useful fallback.
        }

        Path configFile = findUpwards(starts, "cellpose-config.json");
        JsonObject config = readConfig(configFile);
        Path project = null;
        if (config != null && config.has("projectDirectory")) {
            project = resolve(configFile.getParent(), config.get("projectDirectory").getAsString());
        }
        if (project == null) {
            project = findProject(starts);
        }
        if (project == null) {
            throw new IllegalStateException(
                    "Could not find the Cellpose project. Put cellpose-config.json "
                    + "next to the project or open ImageJ from the project directory."
            );
        }

        if (config == null) {
            config = readConfig(project.resolve("cellpose-config.json"));
        }
        String pythonDefault = isWindows()
                ? ".venv/Scripts/python.exe" : ".venv/bin/python";
        Path python = resolve(project, value(config, "pythonPath", pythonDefault));
        Path script = resolve(project, value(config, "predictScript", "predict.py"));
        Path predictions = resolve(project, value(config, "predictionsPath", "predictions.json"));
        return new ProjectSettings(project, python, script, predictions);
    }

    private static Path findProject(List<Path> starts) {
        for (Path start : starts) {
            Path current = Files.isDirectory(start) ? start : start.getParent();
            while (current != null) {
                if (Files.isRegularFile(current.resolve("predict.py"))) {
                    return current.toAbsolutePath().normalize();
                }
                current = current.getParent();
            }
        }
        return null;
    }

    private static Path findUpwards(List<Path> starts, String filename) {
        for (Path start : starts) {
            Path current = Files.isDirectory(start) ? start : start.getParent();
            while (current != null) {
                Path candidate = current.resolve(filename);
                if (Files.isRegularFile(candidate)) return candidate;
                current = current.getParent();
            }
        }
        return null;
    }

    private static JsonObject readConfig(Path file) {
        if (file == null || !Files.isRegularFile(file)) return null;
        try (Reader reader = Files.newBufferedReader(file)) {
            return new Gson().fromJson(reader, JsonObject.class);
        } catch (Exception error) {
            throw new IllegalStateException("Could not read configuration: " + file, error);
        }
    }

    private static String value(JsonObject config, String key, String fallback) {
        return config != null && config.has(key) ? config.get(key).getAsString() : fallback;
    }

    private static Path resolve(Path base, String value) {
        Path path = Paths.get(value);
        return (path.isAbsolute() ? path : base.resolve(path)).toAbsolutePath().normalize();
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().contains("win");
    }
}
