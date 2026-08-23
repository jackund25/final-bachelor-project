import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Glucose Monitor",
    short_name: "Glucose Monitor",
    description:
      "Clinical decision support dashboard for glucose prediction and monitoring.",
    start_url: "/",
    display: "standalone",
    background_color: "#f5f7f9",
    theme_color: "#2f6f8f",
    orientation: "portrait",
    icons: [
      {
        src: "/icons/icon-192.png",
        sizes: "192x192",
        type: "image/png",
      },
      {
        src: "/icons/icon-512.png",
        sizes: "512x512",
        type: "image/png",
      },
    ],
  };
}