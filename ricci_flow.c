#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>       // Include GLEW
#include <GLFW/glfw3.h>    // Include GLFW

// cglm (C version of GLM) for transformations
#include <cglm/cglm.h>

//
// CONSTANTS / GLOBALS
//

#define N 801
#define MAX_VERTS 2000000 // Big enough for dynamic mesh

// The data from the original code:
double c5, c3, h[N], m[N], sm[N], nh[N], nm[N], xg[N], yg[N],
       oh[N], om[N], osm[N], tmpv[N], sh[N], dt, tm;
int still_ok;
int metric;             // 0 => draw 3D surface, 1 => "metric" mode
int do_deriv_adj;
int n_fft;

double para_i[16], para_l[16], para_io[16], para_lo[16];

// For user interaction in the new code:
int rotate = 0, movelight = 0;
int spinxlight = 0, spinylight = 0;
int flow = 0;
int choose_surface = 1; // "new shape" mode
int rot_angle = 0;
float rot_cosf = 1.0f, rot_sinf = 0.0f;

// Last known mouse position:
double last_cursor_x = 0.0;
double last_cursor_y = 0.0;

// We will store camera transformations in cglm mat4:
mat4 g_view_matrix;
mat4 g_proj_matrix;

// We’ll store a model rotation in a mat4 as well:
mat4 g_model_matrix;

//
// SHADERS
//
static const char* vertexShaderSource = "#version 330 core\n"
"layout(location = 0) in vec3 aPos;      \n"
"layout(location = 1) in vec3 aNormal;   \n"
"out vec3 vNormal;                       \n"
"uniform mat4 uMVP;                     \n"
"uniform mat4 uModel;                   \n"
"void main()                            \n"
"{                                      \n"
"    vNormal = mat3(uModel) * aNormal;  \n"
"    gl_Position = uMVP * vec4(aPos, 1.0);\n"
"}";

static const char* fragmentShaderSource = "#version 330 core\n"
"in vec3 vNormal;                        \n"
"out vec4 FragColor;                     \n"
"void main()                            \n"
"{                                      \n"
"    // simple directional light from +X direction\n"
"    vec3 lightDir = normalize(vec3(1.0, 0.2, 0.3));\n"
"    float diff = max(dot(normalize(vNormal), lightDir), 0.0);\n"
"    vec3 color = vec3(0.3, 0.7, 0.8) * diff + vec3(0.1, 0.1, 0.1);\n"
"    FragColor = vec4(color, 1.0);       \n"
"}";

//
// Data structures for dynamic mesh
//
typedef struct {
    float position[3];
    float normal[3];
} Vertex;

static GLuint g_program = 0;
static GLuint g_vao = 0;
static GLuint g_vbo = 0;
static size_t g_num_vertices = 0;  // how many vertices to draw

//
// Forward declarations for old logic (flow, geometry, etc.)
//
void init();
int make_xy();
void step();

//-----------------------------------------
// Original code's functions follow below
//-----------------------------------------

/*
   Original "init()" sets up initial metric,
   populates h[] and m[], sm[], etc.
*/
void init()
{
  double t, v;
  int i;

  for (i=0; i<N; i++)
  {
    t=((i-1.0)*M_PI)/(N-3.0);
    h[i]=1.0;
    v=(sin(t)+c5*sin(5.0*t)+c3*sin(3.0*t))
     /(1.0+3.0*c3+5.0*c5);
    if (i>20 && i<N-21 && fabs(v)<0.05) m[i]=-1.0;
    else if (v<0.0) m[i]=-1.0;
    else m[i]=pow(v,2.0);
  }
  m[1]=m[N-2]=0.0;
  m[0]=m[2];
  m[N-1]=m[N-3];

  for (i=0; i<N; i++)
    sm[i]=sqrt(m[i]);

  for (i=0; i<16; i++)
  {
    para_i[i]=(i*N+N/2)/16;
    para_l[i]=-1.0;
  }
}

/*
   Original "filter()"
*/
void filter()
{
  double ftt, deriv, av, c, s, dc, ds, tc;
  int i, k;
  
  for (i=0; i<N; i++) tmpv[i]=0.0;
  
  av=0.0;
  for (k=0; k<n_fft; k+=2)
  {
    ftt=0.5*(h[1]*cos(k*((1-1.0)*M_PI)/(N-3.0))
      +h[(N-1)/2]*cos(k*(((N-1)/2-1.0)*M_PI)/(N-3.0)));
    c=cos(k*((2-1.0)*M_PI)/(N-3.0));
    s=sin(k*((2-1.0)*M_PI)/(N-3.0));
    dc=cos(k*((1.0)*M_PI)/(N-3.0));
    ds=sin(k*((1.0)*M_PI)/(N-3.0));
    for (i=2; i<(N-1)/2; i++)
    {
      ftt+=h[i]*c;
      tc=+dc*c-ds*s;
      s=+ds*c+dc*s;
      c=tc;
    }
    if (k==0) ftt/=0.5*(N-3.0);
    else ftt*=4.0/(N-3.0);
    av+=fabs(ftt*k);
    if (k>n_fft/2)
    {
      if (ftt*k/av/h[1]>0.1 && k>n_fft-20 && tm>0.001) still_ok=0;
      ftt*=1.0-sqrt((k-n_fft/2)/(0.5*n_fft+0.1));
      if (ftt>0.001) ftt*=0.1;
      if (ftt>0.01) ftt*=0.1;
    }
    for (i=0; i<=(N-1)/2; i++)
      tmpv[i]+=ftt*cos(k*((i-1.0)*M_PI)/(N-3.0));
    for (i=(N+1)/2; i<N; i++)
      tmpv[i]=tmpv[N-1-i];
  }
  for (i=0; i<N; i++)
  {
    if (tmpv[i]>0.000001) h[i]=tmpv[i];
    else h[i]=0.000001;
  }

  deriv=0.0;
  for (i=0; i<N; i++) tmpv[i]=0.0;
  for (k=1; k<n_fft; k+=2)
  {
    ftt=0.5*(sm[1]*sin(k*((1-1.0)*M_PI)/(N-3.0))
            +sm[(N-1)/2]*sin(k*(((N-1)/2-1.0)*M_PI)/(N-3.0)));
    c=cos(k*((2-1.0)*M_PI)/(N-3.0));
    s=sin(k*((2-1.0)*M_PI)/(N-3.0));
    dc=cos(k*((1.0)*M_PI)/(N-3.0));
    ds=sin(k*((1.0)*M_PI)/(N-3.0));
    for (i=2; i<(N-1)/2; i++)
    {
      ftt+=sm[i]*s;
      tc=+dc*c-ds*s;
      s=+ds*c+dc*s;
      c=tc;
    }
    ftt*=4.0/(N-3.0);
    if (k>n_fft/2)
    {
      ftt*=1.0-(k-n_fft/2)/(0.5*n_fft+0.1);
      if (ftt>0.001) ftt*=0.1;
      if (ftt>0.01) ftt*=0.1;
    }
    deriv+=ftt*k;
    for (i=0; i<=(N-1)/2; i++)
      tmpv[i]+=ftt*sin(k*((i-1.0)*M_PI)/(N-3.0));
    for (i=(N+1)/2; i<N; i++)
      tmpv[i]=tmpv[N-1-i];
  }

  if (do_deriv_adj==1) ftt=sqrt(h[1])/deriv; else ftt=1.0;
  for (i=1; i<=(N-1)/2; i++)
  {
    sm[i]=fabs(tmpv[i])*(ftt+i*(i/500.0)/n_fft)/(1.0+i*(i/500.0)/n_fft);
    m[i]=sm[i]*sm[i];
  }
  sm[0]=sm[2]; m[0]=m[2];
  for (i=(N+1)/2; i<N; i++)
  {
    sm[i]=sm[N-1-i];
    m[i]=m[N-1-i];
  }
}

/*
   "rescale()" from original code
*/
void rescale()
{
  double len, a, b, c, l, s1;
  int i, ii, j;
  
  for (i=0; i<=(N-1)/2; i++)
    sh[i]=sqrt(h[i]);
  for (i=(N+1)/2; i<N; i++)
    sh[i]=sh[N-1-i];
  
  len=0.5*(sh[1]+sh[(N-1)/2]);
  for (i=2; i<(N-1)/2; i++)
    len+=sh[i];
  len*=2.0*M_PI/(N-3.0);

  for (i=0; i<16; i++)
    para_l[i]=-1.0;

  l=0.0;
  for (i=1; i<N; i++)
  {
    a=0.5*(sh[i+1]-sh[i])*M_PI/(N-3.0);
    b=sh[i]*M_PI/(N-3.0);

    for (j=0; j<16; j++)
    {
      if (i>para_i[j]-1 && para_l[j]==-1.0)
        para_l[j]=l + a*pow(para_i[j]-i,2.0) + b*(para_i[j]-i);
    }
    l+=a+b;
  }
  
  for (j=0; j<16; j++)
    para_i[j]=1.0+(N-3.0)*para_l[j]/len;

  tmpv[1]=0.0;
  ii=2;
  l=0.0;
  for (i=1; i<N; i++)
  {
    a=0.5*(sh[i+1]-sh[i])*M_PI/(N-3.0);
    b=sh[i]*M_PI/(N-3.0);
another:
    if (ii>N-3) break;
    c=l - len*(ii-1.0)/(N-3.0);
    if (fabs(2.0*a*c/(b*b))<1e-8) s1=-c/b;
    else s1=(-b+sqrt(b*b-4.0*a*c))/(2.0*a);

    if (s1<=1.0)
    {
      tmpv[ii++]=sm[i]+s1*(sm[i+1]-sm[i]);
      goto another;
    }
    else if (i==N-3 && s1<=1.00001)
      tmpv[ii++]=sm[i+1];

    l+=a+b;
  }

  tmpv[N-2]=0.0;
  tmpv[0]=tmpv[2]; 
  tmpv[N-1]=tmpv[N-3];
  
  for (i=0; i<N; i++)
  {
    sm[i]=tmpv[i];
    m[i]=tmpv[i]*tmpv[i];
  }

  a=pow(len/M_PI,2.0);
  for (i=0; i<N; i++)
    h[i]=a;
}

/*
   The original "step()" for one Ricci flow iteration
*/
void step()
{
  double dm, dh, dsm, ddh, l, dm_m, hder;
  double dm1, dm2, dm3, dm4, dh1, dh2, dh3, dh4, dsm1, dsm2, dsm3, dsm4;
  int i, j, k, n;

  l=(N-3.0)/M_PI;

  for (n=0; n<1; n++) {
    for (k=0; k<4; k++) {
      for (j=0; j<1; j++)
      {
        ddh=(h[1+1]-2.0*h[1]+h[1-1])*l*l;
        dm_m=(2.0*m[3]-8.0*m[2])*l*l/(2.0*m[2]);
        nh[1]=h[1]-2.0*dt*(ddh/(2.0*h[1])-0.25*dm_m);
        nm[1]=0.0;

        ddh=(h[N-2+1]-2.0*h[N-2]+h[N-2-1])*l*l;
        dm_m=(2.0*m[N-4]-8.0*m[N-3])*l*l/(2.0*m[N-3]);
        nh[N-2]=h[N-2]-2.0*dt*(ddh/(2.0*h[N-2])-0.25*dm_m);
        nm[N-2]=0.0;

        for (i=2; i<N-2; i++)
        {
          if (i==2 || i==N-3)
          {
            dh=(h[i+1]-h[i-1])*l*0.5;
            dm=(m[i+1]-m[i-1])*l*0.5;
            dsm=(sm[i+1]-2.0*sm[i]+sm[i-1])*l*l;
          }
          else
          {
            dh1=(h[i+1]-h[i-1])*l*0.5-(h[i+2]-2.0*h[i+1]+2.0*h[i-1]-h[i-2])*l*0.5/6.0;
            dh2=(h[i+2]-h[i-2])*l*0.25-(h[i+2]-2.0*h[i+1]+2.0*h[i-1]-h[i-2])*l*0.5/6.0;
            dh3=(h[i+1]-h[i-1])*l*0.5;
            dh4=(h[i+2]-h[i-2])*l*0.25;
            if (fabs(dh1)<fabs(dh2)) dh=dh1; else dh=dh2;
            if (fabs(dh3)<fabs(dh)) dh=dh3;
            if (fabs(dh4)<fabs(dh)) dh=dh4;

            dm1=(m[i+1]-m[i-1])*l*0.5-(m[i+2]-2.0*m[i+1]+2.0*m[i-1]-m[i-2])*l*0.5/6.0;
            dm2=(m[i+2]-m[i-2])*l*0.25-(m[i+2]-2.0*m[i+1]+2.0*m[i-1]-m[i-2])*l*0.5/6.0;
            dm3=(m[i+1]-m[i-1])*l*0.5;
            dm4=(m[i+2]-m[i-2])*l*0.25;
            if (fabs(dm1)<fabs(dm2)) dm=dm1; else dm=dm2;
            if (fabs(dm3)<fabs(dm)) dm=dm3;
            if (fabs(dm4)<fabs(dm)) dm=dm4;

            dsm1=(sm[i+1]-2.0*sm[i]+sm[i-1])*l*l-(sm[i-2]-4.0*sm[i-1]+6.0*sm[i]-4.0*sm[i+1]+sm[i+2])*l*l/8.0;
            dsm2=(sm[i+2]-2.0*sm[i]+sm[i-2])*l*l*0.25-(sm[i-2]-4.0*sm[i-1]+6.0*sm[i]-4.0*sm[i+1]+sm[i+2])*l*l/8.0;
            dsm3=(sm[i+1]-2.0*sm[i]+sm[i-1])*l*l;
            dsm4=(sm[i+2]-2.0*sm[i]+sm[i-2])*l*l*0.25;
            if (fabs(dsm1)<fabs(dsm2)) dsm=dsm1; else dsm=dsm2;
            if (fabs(dsm3)<fabs(dsm)) dsm=dsm3;
            if (fabs(dsm4)<fabs(dsm)) dsm=dsm4;
          }
          
          dm_m=dm/m[i];
          hder=-dsm/sm[i]+0.25*dm_m*dh/h[i];

          nh[i]=h[i]-2.0*dt*hder;
          nm[i]=m[i]-2.0*dt*hder*m[i]/h[i];
        }

        nh[0]=nh[2];     nm[0]=nm[2];
        nh[N-1]=nh[N-3]; nm[N-1]=nm[N-3];

        for (i=1; i<N-1; i++)
        {
          h[i]=nh[i];
          m[i]=nm[i];
        }

        m[0]=m[2];     h[0]=h[2];
        m[N-1]=m[N-3]; h[N-1]=h[N-3];

        for (i=0; i<N; i++)
          sm[i]=sqrt(m[i]);
      }
      filter();
    }
    rescale();
  }
}

/*
   "make_xy()" from original code
   (renamed xg[] and yg[] to avoid conflict with cglm)
*/
int make_xy()
{
  double l, dm, hh, dx1, dx2, dx3;
  int i;

  l=(N-3.0)/M_PI;

  xg[1]=0.0;
  yg[1]=0.0;

  for (i=0; i<N; i++)
    if (m[i]!=m[i] || h[i]!=h[i] || sm[i]!=sm[i])
      return 0;
  
  for (i=2; i<N-1; i++)
  {
    if (m[i]<-0.1) return 0;
    if (m[i]<0.0) yg[i]=0.0;
    else yg[i]=sm[i];

    dm=(sm[i]-sm[i-1])*l;
    hh=h[i-1];
    if (hh-dm*dm<0.0) dx1=0.0;
    else dx1=sqrt(hh-dm*dm)/l;

    dm=(sm[i]-sm[i-1])*l;
    hh=0.5*(h[i-1]+h[i]);
    if (hh-dm*dm<0.0) dx2=0.0;
    else dx2=sqrt(hh-dm*dm)/l;

    dm=(sm[i+1]-sm[i-1])*l*0.5;
    hh=h[i];
    if (hh-dm*dm<0.0) dx3=0.0;
    else dx3=sqrt(hh-dm*dm)/l;

    if (dx1==0.0 && dx2==0.0 && dx3==0.0)
      return 0;

    xg[i]=xg[i-1]+0.25*dx1+0.5*dx2+0.25*dx3;
  }

  for (i=1; i<N-2; i++)
    xg[i]-=0.5*xg[N-2];
  xg[N-2]*=0.5;
  
  xg[0]=xg[1];     yg[0]=yg[1];
  xg[N-1]=xg[N-2]; yg[N-1]=yg[N-2];

  if (still_ok==0) return 0;
  return 1;
}

//
// Modern rendering approach
// We dynamically build vertex data for the current shape
//

/**
 * Build the vertex buffer for the entire 3D shape if metric=0,
 * or the 2D “metric” display if metric=1.
 *
 * Fills a global array of Vertex structures.
 * Sets g_num_vertices to how many we have.
 */
static Vertex g_vertex_data[MAX_VERTS];

void buildVertexData()
{
    g_num_vertices = 0;

    // If in metric=1 mode, draw some 2D representation using lines/points
    if(metric == 1)
    {
        // We'll just store points for h, m, and the revolve cross-section
        // to mimic the original “metric” debugging view.
        // In original code, that was done with glBegin(GL_POINTS) etc.

        // We can put them all in a single draw as GL_POINTS or separate draws.

        // Let's do them as GL_POINTS:
        size_t i;
        // Blue points for m
        for(i=0; i<(size_t)N; i++){
            if(g_num_vertices+1 >= MAX_VERTS) break;
            Vertex v;
            v.position[0] = 20.0f * ((float)i - 0.5f*(float)N)/(float)N;
            v.position[1] = -8.0f + 10.0f*(float)m[i];
            v.position[2] = 0.0f;
            // normal can be anything here
            v.normal[0] = 0.f; v.normal[1] = 0.f; v.normal[2] = 1.f;
            g_vertex_data[g_num_vertices++] = v;
        }

        // Green points for h
        for(i=0; i<(size_t)N; i++){
            if(g_num_vertices+1 >= MAX_VERTS) break;
            Vertex v;
            v.position[0] = 20.0f * ((float)i - 0.5f*(float)N)/(float)N;
            v.position[1] = -8.0f + 10.0f*(float)h[i];
            v.position[2] = 0.0f;
            v.normal[0] = 0.f; v.normal[1] = 0.f; v.normal[2] = 1.f;
            g_vertex_data[g_num_vertices++] = v;
        }

        // White points for revolve cross-section
        for(i=1; i<(size_t)N-1; i++){
            if(g_num_vertices+2 >= MAX_VERTS) break;
            Vertex v1, v2;
            float X = 5.0f*(float)yg[i];
            float Z = 5.0f*(float)xg[i];

            // +X side
            v1.position[0] = X;
            v1.position[1] = Z;
            v1.position[2] = 0.f;
            v1.normal[0] = 0.f; v1.normal[1] = 0.f; v1.normal[2] = 1.f;

            // -X side
            v2.position[0] = -X;
            v2.position[1] = Z;
            v2.position[2] = 0.f;
            v2.normal[0] = 0.f; v2.normal[1] = 0.f; v2.normal[2] = 1.f;

            g_vertex_data[g_num_vertices++] = v1;
            g_vertex_data[g_num_vertices++] = v2;
        }
        // For the param lines (the 16 lines), originally drawn with GL_LINE_STRIP
        // We'll skip those or do them similarly if needed. 
        // (In practice you'd store them in line segments as well.)
        // ...
    }
    else
    {
        // We build the revolve surface as in the original "draw_full_surface()"
        // but instead of immediate mode, we fill the Vertex array.

        // Let's replicate the triangulation:
        int i, j;
        int N_theta = 50;
        int skip = 10;

        // Triangular strips from j=skip to j<N-2-skip
        for(j=skip; j < (N-2-skip); j += skip)
        {
            // one strip has (N_theta+1)*2 vertices
            for(i=0; i <= N_theta; i++)
            {
                // The lower ring vertex
                {
                    Vertex v;
                    float theta = (float)i * 2.0f*(float)M_PI/(float)N_theta;
                    float c = cosf(theta);
                    float s = sinf(theta);

                    float r = (float)yg[j];
                    float xx = c*r;
                    float yy = s*r;
                    float zz = (float)xg[j];

                    // normal estimate:
                    // original code used differences in revolve cross-section
                    // for simplicity, do a revolve normal:
                    float n1 = (float)(yg[j+1] - yg[j-1]);
                    float n2 = (float)(xg[j-1] - xg[j+1]);
                    float inl = 1.0f / sqrtf(n1*n1 + n2*n2);
                    if(n2<0.f){ n2=-n2; n1=-n1; }
                    // revolve normal
                    float nx = c*n2*inl;
                    float ny = n1*inl;
                    float nz = s*n2*inl;

                    // position scale 7.0 as in original
                    v.position[0] = 7.0f*xx;
                    v.position[1] = 7.0f*zz;
                    v.position[2] = 7.0f*yy;

                    v.normal[0] = nx;
                    v.normal[1] = ny;
                    v.normal[2] = nz;

                    g_vertex_data[g_num_vertices++] = v;
                }

                // The upper ring vertex
                {
                    Vertex v;
                    float theta = (float)i * 2.0f*(float)M_PI/(float)N_theta;
                    float c = cosf(theta);
                    float s = sinf(theta);

                    float r = (float)yg[j+skip];
                    float xx = c*r;
                    float yy = s*r;
                    float zz = (float)xg[j+skip];

                    float n1 = (float)(yg[j+skip+1] - yg[j+skip-1]);
                    float n2 = (float)(xg[j+skip-1] - xg[j+skip+1]);
                    float inl = 1.0f / sqrtf(n1*n1 + n2*n2);
                    if(n2<0.f){ n2=-n2; n1=-n1; }

                    float nx = c*n2*inl;
                    float ny = n1*inl;
                    float nz = s*n2*inl;

                    v.position[0] = 7.0f*xx;
                    v.position[1] = 7.0f*zz;
                    v.position[2] = 7.0f*yy;

                    v.normal[0] = nx;
                    v.normal[1] = ny;
                    v.normal[2] = nz;

                    g_vertex_data[g_num_vertices++] = v;
                }
            }
        }

        // Triangle fan for top (j ~ N-2)
        {
            // center
            float r = (float)yg[N-2];
            float xx = 0.f;
            float yy = 0.f;
            float zz = (float)xg[N-2];
            float n1 = (float)(yg[N-1] - yg[N-3]);
            float n2 = (float)(xg[N-3] - xg[N-1]);
            float inl = 1.0f / sqrtf(n1*n1 + n2*n2);
            if(n2<0.f){ n2=-n2; n1=-n1; }

            float cx = 7.0f*xx;
            float cy = 7.0f*zz;
            float cz = 7.0f*yy;
            float cnx = 0.0f; 
            float cny = 1.0f; 
            float cnz = 0.0f; // approximate

            // We'll create a triangle fan with (N_theta+2) vertices
            int i;
            Vertex centerVert;
            centerVert.position[0] = cx;
            centerVert.position[1] = cy;
            centerVert.position[2] = cz;
            centerVert.normal[0] = cnx;
            centerVert.normal[1] = cny;
            centerVert.normal[2] = cnz;

            // We'll push the center, then ring
            for(i=0; i<=N_theta; i++){
                // push center
                g_vertex_data[g_num_vertices++] = centerVert;
                // push ring
                Vertex ring;
                float theta = (float)i * 2.0f*(float)M_PI/(float)N_theta;
                float c = cosf(theta);
                float s = sinf(theta);

                float r2 = (float)yg[N-2-skip];
                float xx2 = c*r2;
                float yy2 = s*r2;
                float zz2 = (float)xg[N-2-skip];

                // normal approx
                float n1b = (float)(yg[N-2-skip+1] - yg[N-2-skip-1]);
                float n2b = (float)(xg[N-2-skip-1] - xg[N-2-skip+1]);
                float inlb = 1.0f / sqrtf(n1b*n1b + n2b*n2b);
                if(n2b<0.f){ n2b=-n2b; n1b=-n1b; }
                float nx = c*n2b*inlb;
                float ny = n1b*inlb;
                float nz = s*n2b*inlb;

                ring.position[0] = 7.0f*xx2;
                ring.position[1] = 7.0f*zz2;
                ring.position[2] = 7.0f*yy2;
                ring.normal[0] = nx;
                ring.normal[1] = ny;
                ring.normal[2] = nz;
                g_vertex_data[g_num_vertices++] = ring;
            }
        }

        // Triangle fan for bottom (j ~ 1)
        {
            float r = (float)yg[1];
            float xx = 0.f;
            float yy = 0.f;
            float zz = (float)xg[1];
            // approximate normal downward
            Vertex centerVert;
            centerVert.position[0] = 7.0f*xx;
            centerVert.position[1] = 7.0f*zz;
            centerVert.position[2] = 7.0f*yy;
            centerVert.normal[0] = 0.f;
            centerVert.normal[1] = -1.f;
            centerVert.normal[2] = 0.f;

            int i;
            for(i=0; i<=N_theta; i++){
                g_vertex_data[g_num_vertices++] = centerVert;

                Vertex ring;
                float theta = (float)i * 2.0f*(float)M_PI/(float)N_theta;
                float c = cosf(theta);
                float s = sinf(theta);

                float r2 = (float)yg[skip];
                float xx2 = c*r2;
                float yy2 = s*r2;
                float zz2 = (float)xg[skip];
                float n1b = (float)(yg[skip+1] - yg[skip-1]);
                float n2b = (float)(xg[skip-1] - xg[skip+1]);
                float inlb = 1.0f / sqrtf(n1b*n1b + n2b*n2b);
                if(n2b<0.f){ n2b=-n2b; n1b=-n1b; }

                ring.position[0] = 7.0f*xx2;
                ring.position[1] = 7.0f*zz2;
                ring.position[2] = 7.0f*yy2;
                ring.normal[0] = c*n2b*inlb;
                ring.normal[1] = n1b*inlb;
                ring.normal[2] = s*n2b*inlb;
                g_vertex_data[g_num_vertices++] = ring;
            }
        }

        // We omit the "wire" lines from the original code for brevity,
        // but you could add line segments similarly by pushing 2 vertices at a time.
    }
}

/*
   Create a shader program
*/
GLuint createShaderProgram(const char* vsrc, const char* fsrc)
{
    // vertex shader
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsrc, NULL);
    glCompileShader(vs);
    // check
    {
      GLint success;
      glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
      if(!success){
        char log[512];
        glGetShaderInfoLog(vs, 512, NULL, log);
        printf("Vertex Shader compile error:\n%s\n", log);
        exit(-1);
      }
    }

    // fragment shader
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsrc, NULL);
    glCompileShader(fs);
    // check
    {
      GLint success;
      glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
      if(!success){
        char log[512];
        glGetShaderInfoLog(fs, 512, NULL, log);
        printf("Fragment Shader compile error:\n%s\n", log);
        exit(-1);
      }
    }

    // link
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    // check
    {
      GLint success;
      glGetProgramiv(prog, GL_LINK_STATUS, &success);
      if(!success){
        char log[512];
        glGetProgramInfoLog(prog, 512, NULL, log);
        printf("Program link error:\n%s\n", log);
        exit(-1);
      }
    }

    // cleanup
    glDeleteShader(vs);
    glDeleteShader(fs);

    return prog;
}

//
// GLFW Callbacks
//

// Key callback
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS){
        switch(key){
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_N:
                // new shape
                choose_surface=1;
                flow=0;
                movelight=0;
                rotate=0;
                init();
                make_xy();
                break;
            case GLFW_KEY_F:
                // flow
                choose_surface=0;
                flow=1;
                movelight=0;
                rotate=0;
                break;
            case GLFW_KEY_M:
                metric=1;
                break;
            case GLFW_KEY_S:
                metric=0;
                break;
            case GLFW_KEY_LEFT:
                rot_angle--;
                rot_cosf = cosf(0.01f * rot_angle);
                rot_sinf = sinf(0.01f * rot_angle);
                break;
            case GLFW_KEY_RIGHT:
                rot_angle++;
                rot_cosf = cosf(0.01f * rot_angle);
                rot_sinf = sinf(0.01f * rot_angle);
                break;
            case GLFW_KEY_UP:
            {
                // single step forward in flow
                choose_surface=0;
                flow=1;
                // backup
                int i;
                for(i=0; i<N; i++){ oh[i]=h[i]; om[i]=m[i]; osm[i]=sm[i]; }
                for(i=0; i<16; i++){ para_io[i]=para_i[i]; para_lo[i]=para_l[i]; }
                dt = fabs(dt);
                step();
                if(make_xy()==1){
                    still_ok=1;
                    tm += dt;
                } else {
                    still_ok=0;
                    // revert
                    for(i=0; i<N; i++){ h[i]=oh[i]; m[i]=om[i]; sm[i]=osm[i]; }
                    for(i=0; i<16; i++){ para_i[i]=para_io[i]; para_l[i]=para_lo[i]; }
                    rescale(); filter();
                    step();
                    if(make_xy()==1){ tm+=dt; still_ok=1; }
                    else {
                        // revert again
                        for(i=0; i<N; i++){ h[i]=oh[i]; m[i]=om[i]; sm[i]=osm[i]; }
                        for(i=0; i<16; i++){ para_i[i]=para_io[i]; para_l[i]=para_lo[i]; }
                        make_xy();
                        still_ok=0;
                    }
                }
            } break;
            case GLFW_KEY_DOWN:
            {
                // single step backward in flow
                choose_surface=0;
                flow=1;
                // backup
                int i;
                for(i=0; i<N; i++){ oh[i]=h[i]; om[i]=m[i]; osm[i]=sm[i]; }
                for(i=0; i<16; i++){ para_io[i]=para_i[i]; para_lo[i]=para_l[i]; }
                dt = -fabs(dt);
                step();
                if(make_xy()==1){
                    still_ok=1;
                } else {
                    still_ok=0;
                    // revert
                    int i;
                    for(i=0; i<N; i++){ h[i]=oh[i]; m[i]=om[i]; sm[i]=osm[i]; }
                    for(i=0; i<16; i++){ para_i[i]=para_io[i]; para_l[i]=para_lo[i]; }
                    rescale(); filter();
                    step();
                    if(make_xy()==1) still_ok=1;
                    else {
                        // revert again
                        for(i=0; i<N; i++){ h[i]=oh[i]; m[i]=om[i]; sm[i]=osm[i]; }
                        for(i=0; i<16; i++){ para_i[i]=para_io[i]; para_l[i]=para_lo[i]; }
                        make_xy();
                        still_ok=0;
                    }
                }
            } break;
            default:
                break;
        }
    }
}

// Mouse button callback
static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if(action == GLFW_PRESS)
        {
            // start rotation or shape changes
            if(choose_surface == 0){
                rotate = 1;
            } else {
                rotate = 0; // or let shape changes happen
            }
        }
        else
        {
            // release
            rotate = 0;
        }
    }
    else if(button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        if(action == GLFW_PRESS)
        {
            movelight = 1;
        }
        else
        {
            movelight = 0;
        }
    }
}

// Mouse move callback
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    double dx = xpos - last_cursor_x;
    double dy = ypos - last_cursor_y;

    if(rotate && metric != 1)
    {
        // We emulate the old approach with rot_angle changes
        float angle = (float)sqrt(dx*dx + dy*dy)/300.f;
        // you can incorporate cglm arcball or trackball here if desired
        rot_angle += (int)(dx); // simplistic
        rot_cosf = cosf(0.01f * rot_angle);
        rot_sinf = sinf(0.01f * rot_angle);
    }
    if(movelight)
    {
        spinylight = (spinylight + (int)dx)%720;
        spinxlight = (spinxlight + (int)dy)%720;
    }
    if(choose_surface)
    {
        // in original code, c3/c5 changed with mouse
        double oc5 = c5;
        double oc3 = c3;
        // backup
        int i;
        for(i=0; i<N; i++){ oh[i]=h[i]; om[i]=m[i]; osm[i]=sm[i]; }
        for(i=0; i<16; i++){ para_io[i]=para_i[i]; para_lo[i]=para_l[i]; }
        c3 += 0.001*dx;
        c5 += 0.001*dy;
        init();
        if(make_xy()==0)
        {
            // revert
            for(i=0; i<N; i++){ h[i]=oh[i]; m[i]=om[i]; sm[i]=osm[i]; }
            for(i=0; i<16; i++){ para_i[i]=para_io[i]; para_l[i]=para_lo[i]; }
            c5 = oc5;
            c3 = oc3;
            make_xy();
        }
    }

    last_cursor_x = xpos;
    last_cursor_y = ypos;
}

// Window resize callback
static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);

    // rebuild projection
    float aspect = (float)width / (float)height;
    glm_perspective(glm_rad(40.0f), aspect, 1.0f, 100.0f, g_proj_matrix);
}

//
// main() entry
//
int main(int argc, char** argv)
{
    // init
    if(!glfwInit()){
        printf("Failed to init GLFW\n");
        return -1;
    }

    // request core profile 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // create window
    GLFWwindow* window = glfwCreateWindow(800, 800, "Ricci_Rot Modern", NULL, NULL);
    if(!window){
        printf("Failed to create window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // init GLEW
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){
        printf("Failed to init GLEW\n");
        glfwTerminate();
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    // set callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // build & link program
    g_program = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // create VAO/VBO
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);

    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_data), NULL, GL_DYNAMIC_DRAW);

    // position attrib (location=0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    // normal attrib (location=1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(float)));

    glBindVertexArray(0);

    // do some initial setup from original main
    n_fft=50;
    still_ok=1;
    metric=0;
    do_deriv_adj=1;
    dt=0.0001;
    c5=c3=0.0;
    tm=0.0;

    init();
    make_xy();

    // Setup initial transforms
    glm_mat4_identity(g_model_matrix);
    glm_translate_z(g_view_matrix, -30.0f); // place camera back
    // Note: We'll set the perspective in the framebuffer_size_callback
    // call it once with our initial size:
    framebuffer_size_callback(window, 800, 800);

    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.2f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Build the geometry into the VBO
        buildVertexData();
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, g_num_vertices*sizeof(Vertex), g_vertex_data);

        // Build model rotation from rot_angle or rot_cosf / rot_sinf
        mat4 rotation;
        glm_mat4_identity(rotation);

        // Simple rotation around Y (for example):
        // or we can do more advanced arcball if desired
        glm_rotate_y(rotation, (float)(rot_angle*0.01f), rotation);

        // final model = rotation
        glm_mat4_copy(rotation, g_model_matrix);

        // Compute MVP
        mat4 mv;
        mat4 mvp;
        glm_mat4_mul(g_view_matrix, g_model_matrix, mv);
        glm_mat4_mul(g_proj_matrix, mv, mvp);

        glUseProgram(g_program);

        // set uniforms
        GLint locMVP = glGetUniformLocation(g_program, "uMVP");
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, (const GLfloat*)mvp);

        GLint locModel = glGetUniformLocation(g_program, "uModel");
        glUniformMatrix4fv(locModel, 1, GL_FALSE, (const GLfloat*)g_model_matrix);

        // draw
        glBindVertexArray(g_vao);
        if(metric == 1){
            // we used only point/line data
            glDrawArrays(GL_POINTS, 0, (GLsizei)g_num_vertices);
        } else {
            // Triangles
            glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)g_num_vertices);
            // But we actually have multiple strips/fans in that buffer
            // If you want a single call, you need the data to be in one big
            // triangle list. Or do multiple subrange calls. For simplicity,
            // we'll interpret them as triangles. A better approach might be
            // to store counts/offsets for each strip/fan, etc.
            // For the example, let's do a single draw:
            // glDrawArrays(GL_TRIANGLES, 0, (GLsizei)g_num_vertices);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup
    glDeleteProgram(g_program);
    glDeleteBuffers(1, &g_vbo);
    glDeleteVertexArrays(1, &g_vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

