
/*
 * OpenMP CPU Ray Tracer
 *
 * Features:
 *   - Sphere intersections
 *   - Diffuse direct lighting
 *   - Hard shadows
 *   - Recursive mirror reflections
 *   - Supersampling anti-aliasing
 *   - PPM (P6) output
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp raytracer.c -lm -o raytracer
 *
 * Run:
 *   ./raytracer [width] [height] [samples_per_pixel] [output.ppm]
 *
 * Example:
 *   OMP_NUM_THREADS=8 ./raytracer 1920 1080 32 render.ppm
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SPHERES 8
#define MAX_DEPTH 4
#define EPSILON 1e-4

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    Vec3 center;
    double radius;
    Vec3 color;
    double reflectivity;
} Sphere;

typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

typedef struct {
    double t;
    Vec3 point;
    Vec3 normal;
    Vec3 color;
    double reflectivity;
} Hit;

static const Vec3 LIGHT_DIRECTION = {-0.6, 1.0, -0.7};

static Sphere spheres[] = {
    {{ 0.0, -0.15, -3.4}, 0.85, {0.85, 0.18, 0.12}, 0.25},
    {{-1.25, -0.35, -4.2}, 0.65, {0.15, 0.35, 0.90}, 0.55},
    {{ 1.15, -0.40, -4.0}, 0.60, {0.15, 0.80, 0.35}, 0.15},
    {{ 0.0, -1001.0, -4.0}, 1000.0, {0.72, 0.72, 0.72}, 0.05}
};

static const int sphere_count = sizeof(spheres) / sizeof(spheres[0]);

static Vec3 v3(double x, double y, double z) {
    return (Vec3){x, y, z};
}

static Vec3 add(Vec3 a, Vec3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static Vec3 sub(Vec3 a, Vec3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static Vec3 mul(Vec3 a, double s) {
    return v3(a.x * s, a.y * s, a.z * s);
}

static Vec3 hadamard(Vec3 a, Vec3 b) {
    return v3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 normalize(Vec3 a) {
    double len = sqrt(dot(a, a));
    return len > 0.0 ? mul(a, 1.0 / len) : v3(0, 0, 0);
}

static Vec3 reflect(Vec3 direction, Vec3 normal) {
    return sub(direction, mul(normal, 2.0 * dot(direction, normal)));
}

static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

/* A tiny deterministic per-pixel random generator.
   Each pixel/sample has its own seed, so threads share no RNG state. */
static uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static double random01(uint32_t *state) {
    *state = hash_u32(*state + 0x9e3779b9U);
    return (double)(*state) / 4294967296.0;
}

static int intersect_sphere(const Ray *ray, const Sphere *sphere,
                            double t_min, double t_max, Hit *hit) {
    Vec3 oc = sub(ray->origin, sphere->center);
    double a = dot(ray->direction, ray->direction);
    double half_b = dot(oc, ray->direction);
    double c = dot(oc, oc) - sphere->radius * sphere->radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0) return 0;

    double root = sqrt(discriminant);
    double t = (-half_b - root) / a;
    if (t < t_min || t > t_max) {
        t = (-half_b + root) / a;
        if (t < t_min || t > t_max) return 0;
    }

    hit->t = t;
    hit->point = add(ray->origin, mul(ray->direction, t));
    hit->normal = normalize(sub(hit->point, sphere->center));
    hit->color = sphere->color;
    hit->reflectivity = sphere->reflectivity;
    return 1;
}

static int closest_hit(const Ray *ray, double t_min, double t_max, Hit *hit) {
    Hit candidate;
    int found = 0;
    double closest = t_max;

    for (int i = 0; i < sphere_count; ++i) {
        if (intersect_sphere(ray, &spheres[i], t_min, closest, &candidate)) {
            found = 1;
            closest = candidate.t;
            *hit = candidate;
        }
    }
    return found;
}

static int is_in_shadow(Vec3 point, Vec3 normal, Vec3 light_dir) {
    Ray shadow_ray = {
        add(point, mul(normal, EPSILON)),
        light_dir
    };
    Hit shadow_hit;
    return closest_hit(&shadow_ray, EPSILON, 1e30, &shadow_hit);
}

static Vec3 sky_color(Vec3 direction) {
    double t = 0.5 * (direction.y + 1.0);
    Vec3 bottom = v3(0.95, 0.96, 1.00);
    Vec3 top = v3(0.35, 0.55, 0.85);
    return add(mul(bottom, 1.0 - t), mul(top, t));
}

static Vec3 trace(const Ray *ray, int depth) {
    Hit hit;

    if (!closest_hit(ray, EPSILON, 1e30, &hit))
        return sky_color(ray->direction);

    Vec3 light_dir = normalize(LIGHT_DIRECTION);
    double diffuse = fmax(0.0, dot(hit.normal, light_dir));
    double visibility = is_in_shadow(hit.point, hit.normal, light_dir) ? 0.0 : 1.0;

    /* Small ambient term prevents completely black shadow regions. */
    double lighting = 0.12 + 0.88 * diffuse * visibility;
    Vec3 local_color = mul(hit.color, lighting);

    if (depth >= MAX_DEPTH || hit.reflectivity <= 0.001)
        return local_color;

    Ray reflected = {
        add(hit.point, mul(hit.normal, EPSILON)),
        normalize(reflect(ray->direction, hit.normal))
    };

    Vec3 reflected_color = trace(&reflected, depth + 1);
    return add(mul(local_color, 1.0 - hit.reflectivity),
               mul(reflected_color, hit.reflectivity));
}

static void write_pixel(FILE *file, Vec3 color) {
    /* Reinhard tone mapping followed by gamma correction. */
    color.x = color.x / (1.0 + color.x);
    color.y = color.y / (1.0 + color.y);
    color.z = color.z / (1.0 + color.z);

    color.x = pow(clamp01(color.x), 1.0 / 2.2);
    color.y = pow(clamp01(color.y), 1.0 / 2.2);
    color.z = pow(clamp01(color.z), 1.0 / 2.2);

    unsigned char rgb[3] = {
        (unsigned char)(255.999 * color.x),
        (unsigned char)(255.999 * color.y),
        (unsigned char)(255.999 * color.z)
    };
    fwrite(rgb, 1, 3, file);
}

int main(int argc, char **argv) {
    int width = 1280;
    int height = 720;
    int samples = 16;
    const char *output = "render.ppm";

    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) samples = atoi(argv[3]);
    if (argc > 4) output = argv[4];

    if (width <= 0 || height <= 0 || samples <= 0) {
        fprintf(stderr, "Width, height, and samples must be positive.\n");
        return EXIT_FAILURE;
    }

    FILE *file = fopen(output, "wb");
    if (!file) {
        perror("Could not open output file");
        return EXIT_FAILURE;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    const double aspect_ratio = (double)width / height;
    const double viewport_height = 2.0;
    const double viewport_width = aspect_ratio * viewport_height;
    const double focal_length = 1.0;
    const Vec3 camera_origin = {0.0, 0.0, 0.0};

    unsigned char *pixels = malloc((size_t)width * height * 3);
    if (!pixels) {
        fprintf(stderr, "Could not allocate image buffer.\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec3 accumulated = v3(0, 0, 0);

            for (int s = 0; s < samples; ++s) {
                uint32_t seed = hash_u32(
                    (uint32_t)(y * width + x) * 9781U +
                    (uint32_t)s * 6271U + 1U
                );

                double jitter_x = random01(&seed);
                double jitter_y = random01(&seed);

                double u = ((x + jitter_x) / width - 0.5) * viewport_width;
                double v = (0.5 - (y + jitter_y) / height) * viewport_height;

                Vec3 direction = normalize(v3(
                    u,
                    v,
                    -focal_length
                ));

                Ray ray = {camera_origin, direction};
                accumulated = add(accumulated, trace(&ray, 0));
            }

            Vec3 pixel = mul(accumulated, 1.0 / samples);
            size_t index = ((size_t)y * width + x) * 3;

            /* Store RGB bytes; each thread owns a unique pixel. */
            pixel.x = pow(clamp01(pixel.x / (1.0 + pixel.x)), 1.0 / 2.2);
            pixel.y = pow(clamp01(pixel.y / (1.0 + pixel.y)), 1.0 / 2.2);
            pixel.z = pow(clamp01(pixel.z / (1.0 + pixel.z)), 1.0 / 2.2);

            pixels[index + 0] = (unsigned char)(255.999 * pixel.x);
            pixels[index + 1] = (unsigned char)(255.999 * pixel.y);
            pixels[index + 2] = (unsigned char)(255.999 * pixel.z);
        }
    }

    double elapsed = omp_get_wtime() - start_time;

    size_t image_size = (size_t)width * height * 3;
    size_t written = fwrite(pixels, 1, image_size, file);

    free(pixels);
    fclose(file);

    if (written != image_size) {
        fprintf(stderr, "Error while writing image.\n");
        return EXIT_FAILURE;
    }

    printf("Rendered %dx%d at %d samples/pixel\n", width, height, samples);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Time: %.3f seconds\n", elapsed);
    printf("Output: %s\n", output);

    return EXIT_SUCCESS;
}
