import numpy as np
import laspy
import scipy as sp
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys




#sets up the matplotlib stuff to display things
fig = plt.figure(
    figsize=(5, 5),
)
graph = fig.add_subplot(111,projection='3d')
graph.set_box_aspect((1,1,1))
graph.set_aspect('equal')
graph.set_xlabel("x (mm)")
graph.set_ylabel("y (mm)")
graph.set_xlabel("z (mm)")
#creates an origin
graph.scatter(0,0,0, color = "black", label = "Origin")





class pointCloud:
    def __init__(self, filename, fid = 1000, floorHeight = -5, x_bounds=None, y_bounds=None):
        #initializes the class with the file and fidelity, and scales
        self.fid = fid
        #loads the LAZ file
        self.data = laspy.read(filename)
        self.name = filename.replace(".laz","")

        #creates a mask to include or uninclude data based on x and y bounds
        mask = np.ones(len(self.data.x), dtype=bool)
        
        if x_bounds:
            mask &= (self.data.x >= x_bounds[0]) & (self.data.x <= x_bounds[1])
        if y_bounds:
            mask &= (self.data.y >= y_bounds[0]) & (self.data.y <= y_bounds[1])

        #moves all the points back towards zero
        x_local = self.data.x - np.min(self.data.x)
        y_local = self.data.y - np.min(self.data.y)
        z_local = self.data.z - np.min(self.data.z)
        
        self.floorHeight = floorHeight

        #retrieves only the xyz coordinates of the file found at multiples of the fidelity.
        #AKA if the fidelited is 20 then it retrieves only every 20th point of the list
        #ie, higher fidelity means less points.
        self.x = np.asarray((x_local)[mask][::self.fid])
        self.y = np.asarray((y_local)[mask][::self.fid])
        self.z = np.asarray((z_local)[mask][::self.fid])
        
        self.count = len(self.x)
        self.tricount = 0


    def print(self):
        try: 
            print(f"Filename: {self.name}\nAmount of points: {self.count}\nAmount of triangles: {self.tricount}\n\tSurface triangles: {self.surfacecount}\n\tSide triangles: {self.sidecount}\n\tfloor triangles: {self.floorcount}")
        except:
            print(f"Filename: {self.name}\nAmount of points: {self.count}\nNo mesh found.")

    def scale(self, scalar):
        self.x = self.x *scalar
        self.y = self.y *scalar
        self.z = self.z *scalar

    def transpose(self):
        #transpose
        temp = self.x
        self.x = self.y
        self.y = temp

    def flip(self, axis):
        #flips the x and y axes
        if axis == "x":
            self.x = np.max(self.x) + (-1 * self.x)
        if axis == "y":
            self.y = np.max(self.y) + (-1 * self.y)


    def append(self, cloud, direction, equalizeHeight = True):
        #allows for height equalization
        heightDiff = np.min(self.z) - np.min(cloud.z)
        if direction == "x":
            self.x = np.append(self.x, (cloud.x + np.max(self.x)))
            self.y = np.append(self.y, (cloud.y))
            if equalizeHeight:
                self.z = np.append(self.z, (cloud.z + heightDiff)) #moves the cloud down to match the other one.
            else:
                self.z = np.append(self.z, cloud.z)
        if direction == "y":
            self.x = np.append(self.x, (cloud.x))
            self.y = np.append(self.y, (cloud.y + np.max(self.y)))
            if equalizeHeight:
                self.z = np.append(self.z, (cloud.z + heightDiff)) #moves the cloud down to match the other one.
            else:
                self.z = np.append(self.z, cloud.z)
        self.count = len(self.x)

    def reset_view(self):
        self.bounds = (self.x.min(), self.x.max())
        graph.set_xlim(self.bounds)
        graph.set_ylim(self.bounds)
        graph.set_zlim(self.bounds)
    
    def plot_points(self, colormap = "viridis", size = .1):
        #plots the points
        self.colors = self.z
        self.colormap = colormap
        scatterplot = graph.scatter(self.x,self.y,self.z, c = self.colors, cmap = self.colormap, s = self.size)
        #plt.colorbar(scatterplot, ax=graph, label='heights')
        self.reset_view()

    def generate_surface(self):
        #smushes the points down to create a 2d 'shadow'
        self.twoDproj = np.vstack([self.x, self.y]).T
        
        #creates a delaunay triangle mesh
        self.Delaunay = Delaunay(self.twoDproj)
        
        #accesses the triangles within the mesh
        self.surfTris = self.Delaunay.simplices
        self.surfacecount = len(self.surfTris)
        self.tricount += self.surfacecount

    def generate_sides(self):
        #accesses the longest convex hull using all the edge points.
        self.hull = self.Delaunay.convex_hull
        self.hullcount = len(self.hull)
        self.sideTris = []
        #for every two points in the hull
        for edge in self.hull:
            p1, p2 = edge
            #sets p1 and p2 to be the indices of where the two points are in the xyz lists
            A = np.array([self.x[p1], self.y[p1], self.z[p1]])
            B = np.array([self.x[p2], self.y[p2], self.z[p2]])

            #smushes the two points down to the floor
            Afloor = np.array([self.x[p1], self.y[p1], self.floorHeight])
            Bfloor = np.array([self.x[p2], self.y[p2], self.floorHeight])

            #creates two triangles, one using two mesh points and one floor point,
            tri1 = np.array([B, A, Bfloor])
            #the other with two floor points and one mesh point
            tri2 = np.array([Afloor, Bfloor, A])
            self.sideTris.append(tri1)
            self.sideTris.append(tri2)
        self.sidecount = len(self.sideTris)
        self.tricount += self.sidecount

    def generate_floor(self):
        self.floorTris = []
        #starts at the first point in the hull
        startindex = self.hull[0][0] 
        startpoint = np.array([self.x[startindex], self.y[startindex], self.floorHeight])
        for edge in self.hull:
            #for every two points in the hull
            i1, i2 = edge
            #creates a triangle between the points and the startpoint
            A = np.array([self.x[i1], self.y[i1], self.floorHeight])
            B = np.array([self.x[i2], self.y[i2], self.floorHeight])
            tri = np.array([startpoint, A,B])
            self.floorTris.append(tri)
        self.floorcount = len(self.floorTris)
        self.tricount += self.floorcount

    def generate_normals(self):
        #creates vectors v1 and v2 along the sides of the triangles
        A = np.stack([self.x[self.surfTris[:, 0]], self.y[self.surfTris[:, 0]], self.z[self.surfTris[:, 0]]], axis=1)
        B = np.stack([self.x[self.surfTris[:, 1]], self.y[self.surfTris[:, 1]], self.z[self.surfTris[:, 1]]], axis=1)
        C = np.stack([self.x[self.surfTris[:, 2]], self.y[self.surfTris[:, 2]], self.z[self.surfTris[:, 2]]], axis=1)

        #Average them to find the center point of each face
        self.face_centers = (A + B + C) / 3.0
        #used to plot them later
        
        v1 = B-A #goes from a to b
        v2 = C-A #goes from a to c

        #creates the normal vectors using the cross products of the sides
        self.normals = np.cross(v1,v2)

        #normalizes every normal vector who's length is not equal to zero
        lengths = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals = np.divide(self.normals, lengths, out=np.zeros_like(self.normals), where=lengths!=0)

    def draw_surface(self):
        #plots the triangular mesh
        graph.plot_trisurf(self.x, self.y, self.z, 
                           triangles = self.surfTris,
                           alpha = 1, linewidth = .01, edgecolor = "black", antialiased=False, shade = False)
        self.reset_view()

    def draw_normals(self):
        viewScale = 1
        graph.quiver(
           self.face_centers[:,0], self.face_centers[:,1], self.face_centers[:,2], #at all A points
           self.normals[:,0] * viewScale, self.normals[:,1] * viewScale, self.normals[:,2] * viewScale, #the components of all the normals
           color = "red"
        )
        self.reset_view()

    def generate_mesh(self):
        print("Generating mesh...")
        print("\tGenerating surface...")
        self.generate_surface()
        print("\tSurface generated!")
        print("\tGenerating sides...")
        self.generate_sides()
        print("\tSides generated!")
        print("\tGenerating floor...")
        self.generate_floor()
        print("\tFloor generated!")
        print("\tGenerating normals...")
        self.generate_normals()
        print("\tNormals generated!")
        self.print()

    def write_surface_facet(self, index, f):
        normal = self.normals[index]

        mag = np.linalg.norm(normal)
        normal = normal / mag if mag != 0 else [0, 0, 0]

        triangle = self.surfTris[index]
        A = ([self.x[triangle[0]], self.y[triangle[0]], self.z[triangle[0]]])
        B = ([self.x[triangle[1]], self.y[triangle[1]], self.z[triangle[1]]])
        C = ([self.x[triangle[2]], self.y[triangle[2]], self.z[triangle[2]]])

        f.write(f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        f.write("  outer loop\n")
        f.write(f"    vertex {A[0]:.6f} {A[1]:.6f} {A[2]:.6f}\n")
        f.write(f"    vertex {B[0]:.6f} {B[1]:.6f} {B[2]:.6f}\n")
        f.write(f"    vertex {C[0]:.6f} {C[1]:.6f} {C[2]:.6f}\n")
        f.write("  endloop\n")
        f.write("endfacet\n")

    def write_custom_facet(self, triangle, normal, f):
        f.write(f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        f.write("  outer loop\n")
        for vert in triangle:
            f.write(f"    vertex {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        f.write("  endloop\n")
        f.write("endfacet\n")

    def create_STL(self):
        print("Beginning file Creation!")
        with open(self.name + ".stl" , "w") as file:
            file.write("solid Body\n") #prints the file header
            count = 0 #used for progress bar
            #writes the surface triangles
            for index in range(self.surfacecount):
                self.write_surface_facet(index, file)
                #prints out percent finished
                sys.stdout.write(f"\r{np.round(((count / self.tricount) * 100), decimals = 1)}% complete ({count + 1}/{self.tricount})")
                sys.stdout.flush()
                count += 1
            #writes the side triangles
            for triangle in self.sideTris:
                self.write_custom_facet(triangle, np.zeros(3), file)
                sys.stdout.write(f"\r{np.round(((count / self.tricount) * 100), decimals = 1)}% complete ({count + 1}/{self.tricount})")
                sys.stdout.flush()
                count += 1
            
            for triangle in self.floorTris:
                self.write_custom_facet(triangle, np.zeros(3), file)
                sys.stdout.write(f"\r{np.round(((count / self.tricount) * 100), decimals = 1)}% complete ({count + 1}/{self.tricount})")
                sys.stdout.flush()
                count += 1
            #ends writing the solid
            file.write("endsolid")
        print("\nExport complete!")




#creates a point cloud from the file
cloud = pointCloud("ExampleLAZ.laz", fid = 2000)

#scales the pointcloud down
cloud.scale((2/10))

#generates the mesh
cloud.generate_mesh()

cloud.draw_surface()
plt.show(block = True)

cloud.create_STL()