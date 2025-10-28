package ecom.intensifier.credibilityengine;

import org.springframework.context.event.EventListener;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;
import org.springframework.boot.context.event.ApplicationReadyEvent;

import java.awt.Desktop;
import java.net.URI;

@Component
public class AutoOpenBrowser {
    private final Environment env;
    public AutoOpenBrowser(Environment env) { this.env = env; }

    @EventListener(ApplicationReadyEvent.class)
    public void openBrowser() {
        try {
            String port = env.getProperty("server.port", "8080");
            String ctx  = env.getProperty("server.servlet.context-path", "/");
            String url  = "http://localhost:" + port + ("/".equals(ctx) ? "/" : ctx + "/");

            if (Desktop.isDesktopSupported()) {
                Desktop.getDesktop().browse(new URI(url));
            } else {
                // Fallback Windows
                new ProcessBuilder("cmd", "/c", "start", url).start();
            }
        } catch (Exception ignored) {}
    }
}
