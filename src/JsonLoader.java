import com.google.gson.Gson;
import com.google.gson.JsonParseException;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

/** Reads a prediction file with Gson. */
public final class JsonLoader {
    private static final Gson GSON = new Gson();

    private JsonLoader() {
    }

    public static PredictionFile load(String path) throws IOException {
        try (Reader reader = new FileReader(path)) {
            PredictionFile predictionFile = GSON.fromJson(reader, PredictionFile.class);

            if (predictionFile == null) {
                throw new JsonParseException("The prediction file is empty.");
            }

            return predictionFile;
        }
    }
}
