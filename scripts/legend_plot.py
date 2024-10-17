import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex')

# Creating and saving a legend using the twilight_shifted colormap ranging from 0 to 180 degrees
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Setting up the colormap and colorbar
cmap = plt.cm.twilight_shifted
norm = plt.Normalize(vmin=0, vmax=180)
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                   cax=ax, orientation='horizontal', label='Azimuth $\\varphi_i$ [deg]')

#cb1.set_label('Azimuth $\\varphi_i$ [deg]', fontsize=18)
#cb1.ax.tick_params(labelsize=12)

# Saving the figure as an EPS file
plt.savefig('./azimuth_colorbar_shifted.svg', format='svg', bbox_inches='tight')
plt.show()
